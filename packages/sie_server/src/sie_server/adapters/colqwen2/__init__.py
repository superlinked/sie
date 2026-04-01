"""ColQwen2.5 adapter for visual document retrieval.

This adapter supports ColQwen2.5-style models that encode document images into
multi-vector representations for late interaction retrieval.

ColQwen2.5 models:
- Take document page images as input
- Return per-patch embeddings (multi-vector) for late interaction scoring
- Also support text query encoding
- Based on Qwen2.5-VL architecture (3B params)

Target model: vidore/colqwen2.5-v0.2

ColQwen2.5 is NOT yet in native transformers. The model class (thin wrapper
around Qwen2_5_VLModel) is inlined here to avoid a colpali-engine dependency
which conflicts with our torch>=2.9 requirement.

Reference: https://github.com/illuin-tech/colpali
See: https://huggingface.co/vidore/colqwen2.5-v0.2
"""

from __future__ import annotations

import gc
import io
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar, Literal

import numpy as np
import torch
from torch import nn

from sie_server.adapters.base import ModelAdapter, ModelCapabilities, ModelDims
from sie_server.core.inference_output import EncodeOutput

if TYPE_CHECKING:
    from PIL import Image

    from sie_server.types.inputs import Item

logger = logging.getLogger(__name__)

ComputePrecision = Literal["float16", "bfloat16", "float32"]

_ERR_NOT_LOADED = "Model not loaded. Call load() first."
_ERR_NO_INPUT = "ColQwen2Adapter requires either text or images input"

# ColQwen2.5 processor constants (from colpali-engine)
_VISUAL_PROMPT_PREFIX = (
    "<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>Describe the image.<|im_end|><|endoftext|>"
)
_QUERY_AUGMENTATION_TOKEN = "<|endoftext|>"  # noqa: S105 — model special token, not a password
_QUERY_AUGMENTATION_COUNT = 10


# ---------------------------------------------------------------------------
# Inlined ColQwen2_5 model class
# (from colpali-engine, avoids dependency conflict with torch>=2.9)
# ---------------------------------------------------------------------------


def _make_colqwen2_5_cls() -> type:
    """Lazily create the ColQwen2_5 model class.

    Defers the transformers import so the module can be loaded without
    transformers installed (for config-only operations).
    """
    from transformers.models.qwen2_5_vl import Qwen2_5_VLConfig, Qwen2_5_VLModel

    class ColQwen2_5(Qwen2_5_VLModel):  # noqa: N801
        """ColQwen2.5: Qwen2.5-VL with ColBERT-style multi-vector projection.

        Adds a linear projection from hidden_size to 128-dim, L2 normalizes,
        and masks with attention_mask. LoRA adapter weights are loaded via peft.

        Inlined from colpali-engine to avoid torch version conflict.
        Reference: https://github.com/illuin-tech/colpali
        """

        main_input_name: ClassVar[str] = "doc_input_ids"

        def __init__(self, config: Qwen2_5_VLConfig) -> None:
            super().__init__(config=config)
            self.dim = 128
            self.custom_text_proj = nn.Linear(self.config.hidden_size, self.dim)
            self.padding_side = "left"
            self.post_init()

        @classmethod
        def from_pretrained(cls, *args: Any, **kwargs: Any) -> ColQwen2_5:
            key_mapping = kwargs.pop("key_mapping", None)
            if key_mapping is None:
                key_mapping = super()._checkpoint_conversion_mapping
            return super().from_pretrained(*args, **kwargs, key_mapping=key_mapping)

        def forward(self, *args: Any, **kwargs: Any) -> torch.Tensor:
            # Handle pixel_values unpadding for DDP compatibility
            if "pixel_values" in kwargs:
                offsets = kwargs["image_grid_thw"][:, 1] * kwargs["image_grid_thw"][:, 2]
                kwargs["pixel_values"] = torch.cat(
                    [pv[:off] for pv, off in zip(kwargs["pixel_values"], offsets)],
                    dim=0,
                )

            kwargs.pop("return_dict", True)
            kwargs.pop("output_hidden_states", None)
            kwargs.pop("use_cache", None)
            last_hidden_states = (
                super()
                .forward(
                    *args,
                    **kwargs,
                    use_cache=False,
                    output_hidden_states=True,
                    return_dict=True,
                )
                .last_hidden_state
            )  # (batch_size, sequence_length, hidden_size)

            proj = self.custom_text_proj(last_hidden_states)  # (batch, seq, 128)
            proj = proj / proj.norm(dim=-1, keepdim=True)  # L2 normalize
            proj = proj * kwargs["attention_mask"].unsqueeze(-1)  # mask padding
            return proj

    return ColQwen2_5


class ColQwen2Adapter(ModelAdapter):
    """Adapter for ColQwen2.5 visual document retrieval models.

    ColQwen2.5 encodes document page images into multi-vector representations
    (one 128-dim vector per image patch) for late interaction retrieval.
    Based on Qwen2.5-VL architecture.
    """

    def __init__(
        self,
        model_name_or_path: str | Path,
        *,
        normalize: bool = True,
        compute_precision: ComputePrecision = "bfloat16",
        trust_remote_code: bool = False,
        max_seq_length: int | None = None,
        muvera_config: dict[str, Any] | None = None,
        token_dim: int = 128,
    ) -> None:
        """Initialize the adapter.

        Args:
            model_name_or_path: HuggingFace model ID or local path.
            normalize: Not used — ColQwen2.5 normalizes internally.
            compute_precision: Compute precision for inference.
            trust_remote_code: Whether to trust remote code.
            max_seq_length: Ignored — ColQwen2.5 uses dynamic sequence length.
            muvera_config: MUVERA config (passed to postprocessor).
            token_dim: Token embedding dimension (fixed at 128).
        """
        self._model_name_or_path = str(model_name_or_path)
        self._normalize = normalize
        self._compute_precision = compute_precision
        self._trust_remote_code = trust_remote_code

        self._model: Any = None
        self._processor: Any = None
        self._device: str | None = None
        self._multivector_dim: int = token_dim

    @property
    def capabilities(self) -> ModelCapabilities:
        return ModelCapabilities(
            inputs=["text", "image"],
            outputs=["multivector", "score"],
        )

    @property
    def dims(self) -> ModelDims:
        return ModelDims(multivector=self._multivector_dim)

    def load(self, device: str) -> None:
        """Load the model onto the specified device."""
        from transformers.models.qwen2_vl.processing_qwen2_vl import Qwen2VLProcessor

        self._device = device

        dtype = self._resolve_dtype(device)
        attn_impl = self._resolve_attn_implementation(device)

        logger.info(
            "Loading ColQwen2.5 model %s on device=%s with dtype=%s, attn=%s",
            self._model_name_or_path,
            device,
            dtype,
            attn_impl,
        )

        # Load processor (Qwen2VLProcessor is the correct base for ColQwen2.5)
        # Reduce max_pixels to limit visual token count for faster inference.
        # Default 1280*28*28 produces 2000-4000 tokens through the 3B decoder;
        # 768*28*28 caps at ~768 tokens for a ~2-3x speedup with modest quality impact.
        self._processor = Qwen2VLProcessor.from_pretrained(
            self._model_name_or_path,
            min_pixels=256 * 28 * 28,
            max_pixels=768 * 28 * 28,
        )
        tokenizer = self._processor.tokenizer  # ty: ignore[unresolved-attribute]
        tokenizer.padding_side = "left"

        # Load model using inlined ColQwen2_5 class
        ColQwen2_5 = _make_colqwen2_5_cls()

        load_kwargs: dict[str, Any] = {
            "torch_dtype": dtype,
            "device_map": device,
        }
        if attn_impl is not None:
            load_kwargs["attn_implementation"] = attn_impl

        self._model = ColQwen2_5.from_pretrained(  # ty: ignore[unresolved-attribute]
            self._model_name_or_path,
            **load_kwargs,
        ).eval()

        self._multivector_dim = getattr(self._model, "dim", 128)

    def _resolve_dtype(self, device: str) -> torch.dtype:
        if not device.startswith("cuda"):
            return torch.float32
        dtype_map = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }
        return dtype_map.get(self._compute_precision, torch.bfloat16)

    def _resolve_attn_implementation(self, device: str) -> str | None:
        """Return "flash_attention_2" if available, else "sdpa"."""
        if not device.startswith("cuda"):
            return None
        try:
            import flash_attn  # ty: ignore[unresolved-import]

            return "flash_attention_2"
        except ImportError:
            logger.info("flash_attn not available, using sdpa attention")
            return "sdpa"

    def unload(self) -> None:
        device = self._device

        if self._model is not None:
            del self._model
            self._model = None
        if self._processor is not None:
            del self._processor
            self._processor = None
        self._device = None

        gc.collect()
        if device and device.startswith("cuda"):
            torch.cuda.empty_cache()
        elif device == "mps":
            torch.mps.empty_cache()

    # ------------------------------------------------------------------
    # Encode
    # ------------------------------------------------------------------

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
        if self._model is None or self._processor is None:
            raise RuntimeError(_ERR_NOT_LOADED)
        self._validate_output_types(output_types)

        # Separate text-only queries from image items for batched processing
        if is_query:
            multivector_list = [self._encode_text(item.text) for item in items if item.text is not None]
        else:
            # Collect all image items for batched encoding
            image_items: list[Item] = []
            text_items: list[Item] = []
            for item in items:
                has_images = item.images is not None and len(item.images) > 0
                if has_images:
                    image_items.append(item)
                elif item.text is not None:
                    text_items.append(item)
                else:
                    raise ValueError(_ERR_NO_INPUT)

            multivector_list: list[np.ndarray] = []
            if image_items:
                all_images: list[Image.Image] = []
                for item in image_items:
                    all_images.extend(self._load_images(item))
                multivector_list.extend(self._encode_images_batched(all_images))
            for item in text_items:
                assert item.text is not None
                multivector_list.append(self._encode_text(item.text))

        return EncodeOutput(
            multivector=multivector_list,
            batch_size=len(items),
            is_query=is_query,
            multivector_token_dim=self._multivector_dim,
        )

    def _encode_single_item(self, item: Any, *, is_query: bool) -> np.ndarray:
        has_text = item.text is not None
        has_images = item.images is not None and len(item.images) > 0

        if not has_text and not has_images:
            raise ValueError(_ERR_NO_INPUT)

        if is_query and has_text:
            return self._encode_text(item.text)
        if has_images:
            return self._encode_images(self._load_images(item))
        if has_text:
            return self._encode_text(item.text)

        raise ValueError(_ERR_NO_INPUT)

    # ------------------------------------------------------------------
    # Image encoding
    # ------------------------------------------------------------------

    def _load_images(self, item: Any) -> list[Image.Image]:
        from PIL import Image

        pil_images = []
        for img_input in item.images or []:
            pil_img = Image.open(io.BytesIO(img_input["data"]))
            if pil_img.mode != "RGB":
                pil_img = pil_img.convert("RGB")
            pil_images.append(pil_img)
        return pil_images

    def _encode_images(self, images: list[Image.Image]) -> np.ndarray:
        """Encode images using ColQwen2.5 visual prompt pattern."""
        assert self._model is not None
        assert self._processor is not None

        # Process images with visual prompt prefix (replicates ColQwen2_5_Processor.process_images)
        batch = self._processor(
            text=[_VISUAL_PROMPT_PREFIX] * len(images),
            images=images,
            padding="longest",
            return_tensors="pt",
        )

        # Pad pixel_values per-image for batched inference
        # (handles variable number of patches per image)
        offsets = batch["image_grid_thw"][:, 1] * batch["image_grid_thw"][:, 2]
        pixel_values = list(torch.split(batch["pixel_values"], offsets.tolist()))
        batch["pixel_values"] = torch.nn.utils.rnn.pad_sequence(
            pixel_values,
            batch_first=True,
        )

        batch = {k: v.to(self._device) for k, v in batch.items()}

        with torch.inference_mode():
            embeddings = self._model(**batch)  # (batch, seq, 128)

        if len(images) == 1:
            result = embeddings[0].float().cpu().numpy()
        else:
            result = np.concatenate(
                [embeddings[i].float().cpu().numpy() for i in range(len(images))],
                axis=0,
            )

        # Free GPU memory from intermediate tensors to prevent OOM on
        # subsequent encode calls (L4 22GB GPUs are tight for VLM models).
        del embeddings, batch
        if self._device and self._device.startswith("cuda"):
            torch.cuda.empty_cache()

        return result

    def _encode_images_batched(self, images: list[Image.Image]) -> list[np.ndarray]:
        """Encode multiple images in a single batched forward pass."""
        assert self._model is not None
        assert self._processor is not None

        # Process all images with visual prompt prefix
        batch = self._processor(
            text=[_VISUAL_PROMPT_PREFIX] * len(images),
            images=images,
            padding="longest",
            return_tensors="pt",
        )

        # Pad pixel_values per-image for batched inference
        offsets = batch["image_grid_thw"][:, 1] * batch["image_grid_thw"][:, 2]
        pixel_values = list(torch.split(batch["pixel_values"], offsets.tolist()))
        batch["pixel_values"] = torch.nn.utils.rnn.pad_sequence(
            pixel_values,
            batch_first=True,
        )

        batch = {k: v.to(self._device) for k, v in batch.items()}

        with torch.inference_mode():
            embeddings = self._model(**batch)  # (batch, seq, 128)

        results = [embeddings[i].float().cpu().numpy() for i in range(len(images))]

        del embeddings, batch

        return results

    # ------------------------------------------------------------------
    # Text encoding
    # ------------------------------------------------------------------

    def _encode_text(self, text: str) -> np.ndarray:
        """Encode query text with augmentation suffix."""
        assert self._model is not None
        assert self._processor is not None

        # Add query augmentation tokens (replicates ColQwen2_5_Processor.process_queries)
        augmented = text + _QUERY_AUGMENTATION_TOKEN * _QUERY_AUGMENTATION_COUNT

        batch = self._processor(
            text=[augmented],
            return_tensors="pt",
            padding="longest",
        )
        batch = {k: v.to(self._device) for k, v in batch.items()}

        with torch.inference_mode():
            embeddings = self._model(**batch)  # (1, seq, 128)

        return embeddings[0].float().cpu().numpy()

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------

    def score(
        self,
        query: Any,
        items: list[Any],
        *,
        instruction: str | None = None,
        options: dict[str, Any] | None = None,
    ) -> list[float]:
        """Score documents against a text query using MaxSim."""
        if self._model is None:
            raise RuntimeError(_ERR_NOT_LOADED)

        query_output = self.encode([query], output_types=["multivector"], is_query=True)
        if query_output.multivector is None:
            raise RuntimeError("Failed to encode query: no multivector output")
        query_vecs = query_output.multivector[0]

        doc_output = self.encode(items, output_types=["multivector"], is_query=False)
        if doc_output.multivector is None:
            raise RuntimeError("Failed to encode documents: no multivector output")

        scores = []
        query_tensor = torch.from_numpy(query_vecs).to(self._device)
        for doc_vecs in doc_output.multivector:
            doc_tensor = torch.from_numpy(doc_vecs).to(self._device)
            sim = torch.matmul(query_tensor, doc_tensor.T)
            scores.append(sim.max(dim=-1).values.sum().item())
        return scores

    def _validate_output_types(self, output_types: list[str]) -> None:
        unsupported = set(output_types) - {"multivector"}
        if unsupported:
            msg = f"Unsupported output types: {unsupported}. ColQwen2Adapter only supports 'multivector'."
            raise ValueError(msg)

    def get_preprocessor(self) -> Any | None:
        # ColQwen2.5 handles image processing internally via _encode_images()
        # because Qwen2VLProcessor requires text alongside images (visual prompt prefix).
        # The generic ImagePreprocessor doesn't support this pattern.
        return None
