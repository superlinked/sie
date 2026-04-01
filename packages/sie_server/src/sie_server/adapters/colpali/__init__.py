"""ColPali adapter for visual document retrieval.

This adapter supports ColPali-style models that encode document images into
multi-vector representations for late interaction retrieval.

ColPali models:
- Take document page images as input
- Return per-patch embeddings (multi-vector) for late interaction scoring
- Also support text query encoding for retrieval

Target model: vidore/colpali-v1.3-hf (PaliGemma-based, 3B params)

Per roadmap Project 10.5 Phase 1:
- Uses transformers ColPaliForRetrieval + ColPaliProcessor
- Optimization (FA2 varlen, SGLang) deferred to Phase 2/3

See: https://huggingface.co/vidore/colpali-v1.3-hf
"""

from __future__ import annotations

import gc
import io
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import torch

from sie_server.adapters.base import ModelAdapter, ModelCapabilities, ModelDims
from sie_server.core.inference_output import EncodeOutput

if TYPE_CHECKING:
    from PIL import Image
    from transformers import ColPaliForRetrieval, ColPaliProcessor

    from sie_server.types.inputs import Item

logger = logging.getLogger(__name__)

ComputePrecision = Literal["float16", "bfloat16", "float32"]

_ERR_NOT_LOADED = "Model not loaded. Call load() first."
_ERR_NO_INPUT = "ColPaliAdapter requires either text or images input"


class ColPaliAdapter(ModelAdapter):
    """Adapter for ColPali visual document retrieval models.

    ColPali encodes document page images into multi-vector representations
    (one 128-dim vector per image patch) for late interaction retrieval.
    Also supports text query encoding.

    Uses HuggingFace transformers ColPaliForRetrieval and ColPaliProcessor.
    """

    def __init__(
        self,
        model_name_or_path: str | Path,
        *,
        normalize: bool = True,
        compute_precision: ComputePrecision = "float32",
        trust_remote_code: bool = False,
        max_seq_length: int | None = None,
        muvera_config: dict[str, Any] | None = None,
        token_dim: int | None = None,
    ) -> None:
        """Initialize the adapter.

        Args:
            model_name_or_path: HuggingFace model ID or local path.
            normalize: Whether to L2-normalize embeddings.
            compute_precision: Compute precision for inference.
                Note: ColPali with eager attention may require float32 due to
                attention mask dtype issues. flash_attention_2 allows fp16/bf16.
            trust_remote_code: Whether to trust remote code.
            max_seq_length: Ignored - ColPali uses dynamic sequence length.
            muvera_config: Optional MUVERA configuration for converting
                multi-vector to dense representation. Reserved for future use.
            token_dim: Optional token dimension override.
        """
        self._model_name_or_path = str(model_name_or_path)
        self._normalize = normalize
        self._compute_precision = compute_precision
        self._trust_remote_code = trust_remote_code
        self._muvera_config = muvera_config
        self._token_dim = token_dim

        self._model: ColPaliForRetrieval | None = None
        self._processor: ColPaliProcessor | None = None
        self._device: str | None = None
        self._multivector_dim: int = token_dim or 128  # ColPali uses 128-dim per patch

    @property
    def capabilities(self) -> ModelCapabilities:
        """Return model capabilities."""
        return ModelCapabilities(
            inputs=["text", "image"],
            outputs=["multivector", "score"],
        )

    @property
    def dims(self) -> ModelDims:
        """Return model dimensions."""
        return ModelDims(multivector=self._multivector_dim)

    def load(self, device: str) -> None:
        """Load the model onto the specified device.

        Args:
            device: Device string (e.g., "cuda:0", "cpu").
        """
        from transformers import ColPaliConfig, ColPaliForRetrieval, ColPaliProcessor

        self._device = device

        # Determine dtype and attention implementation
        dtype = self._resolve_dtype(device)
        attn_impl = self._resolve_attn_implementation(device)

        logger.info(
            "Loading ColPali model %s on device=%s with dtype=%s, attn=%s",
            self._model_name_or_path,
            device,
            dtype,
            attn_impl,
        )

        # Load processor
        self._processor = ColPaliProcessor.from_pretrained(
            self._model_name_or_path,
            trust_remote_code=self._trust_remote_code,
        )

        # WORKAROUND: The ColPali model config on HuggingFace (vidore/colpali-v1.3-hf) has
        # text_config.torch_dtype set to "float32" even though the model uses bfloat16.
        # This causes flash_attention_2 to fail because PaliGemma uses text_config.torch_dtype
        # to create internal tensors (attention masks, etc.).
        # We fix this by loading the config first and overriding the text_config dtype.
        # Upstream bug: https://huggingface.co/vidore/colpali-v1.3-hf/blob/main/config.json
        # (text_config.torch_dtype should be "bfloat16", not "float32")
        config = ColPaliConfig.from_pretrained(
            self._model_name_or_path,
            trust_remote_code=self._trust_remote_code,
        )
        self._fix_config_dtype(config, dtype)

        # Load model with device_map to ensure proper dtype propagation
        self._model = ColPaliForRetrieval.from_pretrained(
            self._model_name_or_path,
            config=config,
            torch_dtype=dtype,
            device_map=device,
            attn_implementation=attn_impl,
            trust_remote_code=self._trust_remote_code,
        ).eval()

        # Get embedding dimension from model config
        # ColPali projects to 128-dim embeddings
        if hasattr(self._model.config, "embedding_dim"):
            self._multivector_dim = self._model.config.embedding_dim
        else:
            # Default ColPali dimension
            self._multivector_dim = 128

        # Cache input_ids/attention_mask for image encoding — ColPali requires
        # text placeholder tokens alongside pixel_values, but these are identical
        # for all 448×448 images. Computing once at load time saves ~15ms per batch.
        from PIL import Image as PILImage

        dummy = PILImage.new("RGB", (448, 448), color="white")
        cached = self._processor(images=[dummy], return_tensors="pt")
        self._cached_input_ids = cached["input_ids"]  # [1, seq_len]
        self._cached_attention_mask = cached["attention_mask"]  # [1, seq_len]

    def _resolve_dtype(self, device: str) -> torch.dtype:
        """Resolve dtype based on device and config."""
        # CPU should use FP32
        if not device.startswith("cuda"):
            return torch.float32

        dtype_map = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }
        return dtype_map.get(self._compute_precision, torch.float32)

    def _resolve_attn_implementation(self, device: str) -> str:
        """Resolve attention implementation based on device and available libraries."""
        if not device.startswith("cuda"):
            return "eager"

        # Use flash_attention_2 if available for best performance
        try:
            import flash_attn  # ty: ignore[unresolved-import]

            return "flash_attention_2"
        except ImportError:
            logger.info("flash_attn not available, using sdpa attention")
            return "sdpa"

    def _fix_config_dtype(self, config: Any, dtype: torch.dtype) -> None:
        """Fix dtype settings throughout the model config.

        VLM model configs on HuggingFace may have incorrect torch_dtype settings
        in various config locations. Flash Attention requires fp16/bf16, so we need
        to ensure all config paths use the correct dtype.

        Config paths that may need fixing:
        - config.torch_dtype (root level)
        - config.text_config.torch_dtype
        - config.vision_config.torch_dtype
        - config.vlm_config.torch_dtype
        - config.vlm_config.text_config.torch_dtype
        - config.vlm_config.vision_config.torch_dtype
        """
        # Use actual torch.dtype objects — PaliGemma's _update_causal_mask
        # calls torch.finfo(self.text_config_dtype) which fails on strings.
        # Fix root-level dtype
        if hasattr(config, "torch_dtype"):
            config.torch_dtype = dtype

        # Fix text_config dtype
        if hasattr(config, "text_config") and config.text_config is not None:
            config.text_config.torch_dtype = dtype

        # Fix vision_config dtype
        if hasattr(config, "vision_config") and config.vision_config is not None:
            if hasattr(config.vision_config, "torch_dtype"):
                config.vision_config.torch_dtype = dtype

        # Fix vlm_config dtype (nested VLM config for PaliGemma)
        if hasattr(config, "vlm_config") and config.vlm_config is not None:
            if hasattr(config.vlm_config, "torch_dtype"):
                config.vlm_config.torch_dtype = dtype

            # Fix vlm_config.text_config dtype
            if hasattr(config.vlm_config, "text_config") and config.vlm_config.text_config is not None:
                config.vlm_config.text_config.torch_dtype = dtype

            # Fix vlm_config.vision_config dtype
            if hasattr(config.vlm_config, "vision_config") and config.vlm_config.vision_config is not None:
                if hasattr(config.vlm_config.vision_config, "torch_dtype"):
                    config.vlm_config.vision_config.torch_dtype = dtype

    def unload(self) -> None:
        """Unload the model and free resources."""
        device = self._device

        if self._model is not None:
            del self._model
            self._model = None

        if self._processor is not None:
            del self._processor
            self._processor = None

        self._device = None

        # Release GPU memory
        gc.collect()
        if device and device.startswith("cuda"):
            torch.cuda.empty_cache()
        elif device == "mps":
            torch.mps.empty_cache()

    def encode(
        self,
        items: list[Item],
        output_types: list[str],
        *,
        instruction: str | None = None,
        is_query: bool = False,
        prepared_items: list[Any] | None = None,
        options: dict[str, Any] | None = None,
    ) -> EncodeOutput:
        """Run inference returning standardized batched output.

        For document images: returns per-patch embeddings (~1030 vectors per page).
        For text queries: returns per-token embeddings.

        Args:
            items: List of items to encode (with text or images).
            output_types: Which outputs to return (only "multivector" supported).
            instruction: Optional instruction (not used by ColPali).
            is_query: Whether items are queries (True) or documents (False).
                For queries, expects text input.
                For documents, expects image input.
            prepared_items: Optional preprocessed items from PreprocessorRegistry.
                If provided with valid pixel_values, uses those instead of reprocessing.

        Returns:
            EncodeOutput with multivector embeddings.
        """
        if self._model is None or self._processor is None:
            raise RuntimeError(_ERR_NOT_LOADED)

        self._validate_output_types(output_types)

        # Check if we have preprocessed items with pixel_values
        if prepared_items and not is_query and self._has_prepared_pixel_values(prepared_items):
            return self._encode_prepared_batch(items, prepared_items, is_query=is_query)

        # Fallback: process items individually (original behavior)
        multivector_list = []
        for item in items:
            embedding = self._encode_single_item(item, is_query=is_query)
            multivector_list.append(embedding)

        return EncodeOutput(
            multivector=multivector_list,
            batch_size=len(items),
            is_query=is_query,
            multivector_token_dim=self._multivector_dim,
        )

    def _encode_single_item(self, item: Any, *, is_query: bool) -> np.ndarray:
        """Encode a single item (text or image).

        Args:
            item: Item with text or images.
            is_query: Whether this is a query (text) or document (image).

        Returns:
            Numpy array of shape [num_tokens, 128].
        """
        # Item is a TypedDict (dict) - no instance check needed
        has_text = item.text is not None
        has_images = item.images is not None and len(item.images) > 0

        if not has_text and not has_images:
            raise ValueError(_ERR_NO_INPUT)

        # For queries, prefer text; for documents, prefer images
        if is_query and has_text:
            return self._encode_text(item.text)
        if has_images:
            pil_images = self._load_images(item)
            return self._encode_images(pil_images)
        if has_text:
            return self._encode_text(item.text)

        raise ValueError(_ERR_NO_INPUT)

    def _load_images(self, item: Any) -> list[Image.Image]:
        """Load images from item into PIL Images.

        Args:
            item: Item with images field.

        Returns:
            List of PIL Images.
        """
        from PIL import Image

        pil_images = []
        for img_input in item.images or []:
            img_bytes = img_input["data"]
            pil_img = Image.open(io.BytesIO(img_bytes))
            # Convert to RGB if necessary
            if pil_img.mode != "RGB":
                pil_img = pil_img.convert("RGB")
            pil_images.append(pil_img)

        return pil_images

    def _encode_images(self, images: list[Image.Image]) -> np.ndarray:
        """Encode document images into multi-vector embeddings.

        Args:
            images: List of PIL Images (document pages).

        Returns:
            Numpy array of shape [num_patches, 128] for first image.
            If multiple images, returns concatenated patches.
        """
        assert self._model is not None
        assert self._processor is not None

        from torch.nn import functional

        # Process images
        batch = self._processor(images=images, return_tensors="pt")
        batch = {k: v.to(self._device) for k, v in batch.items()}

        with torch.inference_mode():
            outputs = self._model(**batch)
            embeddings = outputs.embeddings  # [batch, num_patches, 128]

            # L2 normalize if configured
            if self._normalize:
                embeddings = functional.normalize(embeddings, p=2, dim=-1)

        # For single image, return [num_patches, 128]
        # For multiple images, concatenate patches
        if len(images) == 1:
            return embeddings[0].float().cpu().numpy()

        # Concatenate all image patches
        all_patches = []
        for i in range(len(images)):
            all_patches.append(embeddings[i].float().cpu().numpy())
        return np.concatenate(all_patches, axis=0)

    def _encode_text(self, text: str) -> np.ndarray:
        """Encode text query into multi-vector embeddings.

        Args:
            text: Query text string.

        Returns:
            Numpy array of shape [num_tokens, 128].
        """
        assert self._model is not None
        assert self._processor is not None

        from torch.nn import functional

        # Process text
        batch = self._processor(text=[text], return_tensors="pt")
        batch = {k: v.to(self._device) for k, v in batch.items()}

        with torch.inference_mode():
            outputs = self._model(**batch)
            embeddings = outputs.embeddings  # [1, num_tokens, 128]

            # L2 normalize if configured
            if self._normalize:
                embeddings = functional.normalize(embeddings, p=2, dim=-1)

        return embeddings[0].float().cpu().numpy()

    def _has_prepared_pixel_values(self, prepared_items: list[Any]) -> bool:
        """Check if prepared items have valid pixel_values for batched inference.

        Args:
            prepared_items: List of PreparedItem objects.

        Returns:
            True if all prepared items have pixel_values tensors.
        """
        if not prepared_items:
            return False

        for prepared_item in prepared_items:
            # Check if this is a PreparedItem with an ImagePayload
            payload = getattr(prepared_item, "payload", None)
            if payload is None:
                return False
            pixel_values = getattr(payload, "pixel_values", None)
            if pixel_values is None:
                return False

        return True

    def _encode_prepared_batch(
        self,
        items: list[Any],
        prepared_items: list[Any],
        *,
        is_query: bool,
    ) -> EncodeOutput:
        """Encode a batch of images using preprocessed pixel_values.

        This enables CPU-parallel preprocessing in the PreprocessorRegistry
        thread pool, with batched GPU inference here.

        Args:
            items: Original Item objects (for IDs).
            prepared_items: PreparedItem objects with ImagePayload.pixel_values.
            is_query: Whether items are queries.

        Returns:
            EncodeOutput with multivector embeddings.
        """
        assert self._model is not None
        assert self._processor is not None

        from PIL import Image
        from torch.nn import functional

        # Stack all pixel_values into a batch tensor
        pixel_values_list = []
        for prepared_item in prepared_items:
            pv = prepared_item.payload.pixel_values
            if pv.dim() == 3:
                pv = pv.unsqueeze(0)  # Add batch dimension if needed
            pixel_values_list.append(pv)

        # Concatenate into batch [N, C, H, W]
        batch_pixel_values = torch.cat(pixel_values_list, dim=0)
        batch_pixel_values = batch_pixel_values.to(self._device)

        # ColPali requires input_ids (placeholder tokens for image patches) in addition
        # to pixel_values. Use cached values from load() — these are identical for all
        # 448×448 images. Expand to batch size and move to device.
        batch_size = len(prepared_items)
        input_ids = self._cached_input_ids.expand(batch_size, -1).to(self._device)
        attention_mask = self._cached_attention_mask.expand(batch_size, -1).to(self._device)

        # Run batched inference
        with torch.inference_mode():
            outputs = self._model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values=batch_pixel_values,
            )
            embeddings = outputs.embeddings  # [batch, num_patches, 128]

            # L2 normalize if configured
            if self._normalize:
                embeddings = functional.normalize(embeddings, p=2, dim=-1)

        # Build multivector list for each item
        multivector_list = []
        for i in range(len(items)):
            emb = embeddings[i].float().cpu().numpy()
            multivector_list.append(emb)

        # Free GPU memory from intermediate tensors to prevent OOM on
        # subsequent batches (L4 22GB GPUs are tight for VLM models).
        del embeddings, batch_pixel_values, input_ids, attention_mask
        if self._device and self._device.startswith("cuda"):
            torch.cuda.empty_cache()

        return EncodeOutput(
            multivector=multivector_list,
            batch_size=len(items),
            is_query=is_query,
            multivector_token_dim=self._multivector_dim,
        )

    def score(
        self,
        query: Any,
        items: list[Any],
        *,
        instruction: str | None = None,
        options: dict[str, Any] | None = None,
    ) -> list[float]:
        """Score document images against a text query using MaxSim.

        MaxSim computes the sum of maximum cosine similarities between
        each query token and all document patch embeddings.

        Args:
            query: Query item (with text).
            items: List of document items (with images).
            instruction: Optional instruction (not used).
            options: Optional options (not used).

        Returns:
            List of MaxSim scores, one per document.
        """
        if self._model is None:
            raise RuntimeError(_ERR_NOT_LOADED)

        # Encode query
        query_output = self.encode(
            [query],
            output_types=["multivector"],
            is_query=True,
        )
        if query_output.multivector is None:
            raise RuntimeError("Failed to encode query: no multivector output")
        query_vecs = query_output.multivector[0]  # [num_query_tokens, 128]

        # Encode documents
        doc_output = self.encode(
            items,
            output_types=["multivector"],
            is_query=False,
        )

        # Compute MaxSim for each document
        scores = []
        query_tensor = torch.from_numpy(query_vecs).to(self._device)

        assert doc_output.multivector is not None
        for doc_vecs in doc_output.multivector:
            doc_tensor = torch.from_numpy(doc_vecs).to(self._device)

            # MaxSim: for each query token, find max similarity with any doc patch
            sim = torch.matmul(query_tensor, doc_tensor.T)
            maxsim_score = sim.max(dim=-1).values.sum().item()
            scores.append(maxsim_score)

        return scores

    def _validate_output_types(self, output_types: list[str]) -> None:
        """Validate that output types are supported."""
        unsupported = set(output_types) - {"multivector"}
        if unsupported:
            msg = f"Unsupported output types: {unsupported}. ColPaliAdapter only supports 'multivector'."
            raise ValueError(msg)

    def get_preprocessor(self) -> Any | None:
        """Return an ImagePreprocessor for CPU/GPU overlap.

        Returns:
            ImagePreprocessor wrapping the ColPaliProcessor, or None if not loaded.
        """
        if self._processor is None:
            return None

        from sie_server.core.preprocessor import ImagePreprocessor

        return ImagePreprocessor(self._processor, self._model_name_or_path)
