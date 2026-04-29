"""SigLIP model adapter for image-text embedding.

This adapter provides support for SigLIP (Sigmoid Loss for Language Image Pre-training)
models that produce aligned embeddings for both images and text in a shared vector space.

Per roadmap Project 10.4, uses transformers SiglipModel with SiglipProcessor
for Phase 1. SigLIP differs from CLIP in using sigmoid loss instead of softmax
and not having a separate projection_dim - it uses hidden_size directly.

Supports:
- Text-only encoding → dense embeddings
- Image-only encoding → dense embeddings
- Image+text encoding → image embeddings (for retrieval)

Example configuration:
    SiglipAdapter(
        model_name_or_path="google/siglip-so400m-patch14-384",
    )
"""

from __future__ import annotations

import io
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar

import torch

from sie_server.adapters._base_adapter import BaseAdapter
from sie_server.adapters._spec import AdapterSpec
from sie_server.adapters._types import ERR_NOT_LOADED, ComputePrecision
from sie_server.core.inference_output import EncodeOutput

if TYPE_CHECKING:
    from PIL import Image
    from transformers import SiglipModel, SiglipProcessor

    from sie_server.types.inputs import Item

logger = logging.getLogger(__name__)

# Error messages
_ERR_NO_INPUT = "SiglipAdapter requires either text or images input"


class SiglipAdapter(BaseAdapter):
    """Adapter for SigLIP image-text embedding models.

    Supports encoding text, images, or both into dense embeddings in a shared
    vector space. Uses HuggingFace transformers SiglipModel and SiglipProcessor.

    Key difference from CLIP: SigLIP uses hidden_size directly instead of
    projection_dim for the embedding dimension.
    """

    spec: ClassVar[AdapterSpec] = AdapterSpec(
        inputs=("text", "image"),
        outputs=("dense",),
        unload_fields=("_model", "_processor", "_dense_dim"),
        default_preprocessor="image",
    )

    def __init__(
        self,
        model_name_or_path: str | Path,
        *,
        normalize: bool = True,
        compute_precision: ComputePrecision = "float16",
        trust_remote_code: bool = False,
        max_seq_length: int | None = None,
    ) -> None:
        """Initialize the adapter.

        Args:
            model_name_or_path: HuggingFace model ID or local path.
            normalize: Whether to L2-normalize embeddings.
            compute_precision: Compute precision for inference.
            trust_remote_code: Whether to trust remote code.
            max_seq_length: Ignored - SigLIP uses fixed token length from model config.
        """
        self._model_name_or_path = str(model_name_or_path)
        self._normalize = normalize
        self._compute_precision = compute_precision
        self._trust_remote_code = trust_remote_code

        self._model: SiglipModel | None = None
        self._processor: SiglipProcessor | None = None
        self._device: str | None = None
        self._dense_dim: int | None = None

    def load(self, device: str) -> None:
        """Load the model onto the specified device.

        Args:
            device: Device string (e.g., "cuda:0", "cpu").
        """
        from transformers import SiglipModel, SiglipProcessor

        self._device = device

        # Determine dtype
        dtype = self._resolve_dtype()

        logger.info(
            "Loading SigLIP model %s on device=%s with dtype=%s",
            self._model_name_or_path,
            device,
            dtype,
        )

        # Load processor (handles both text tokenization and image preprocessing)
        self._processor = SiglipProcessor.from_pretrained(
            self._model_name_or_path,
            trust_remote_code=self._trust_remote_code,
        )

        # Load model
        self._model = SiglipModel.from_pretrained(
            self._model_name_or_path,
            torch_dtype=dtype,
            trust_remote_code=self._trust_remote_code,
        )
        self._model.to(device)
        self._model.eval()

        # Get embedding dimension from model config
        # SigLIP uses hidden_size directly (not projection_dim like CLIP)
        # The vision and text encoders should have the same hidden_size
        self._dense_dim = self._model.config.vision_config.hidden_size

    def _resolve_dtype(self) -> torch.dtype:
        """Resolve dtype based on device and config."""
        # CPU should use FP32
        if not self._device or not str(self._device).startswith("cuda"):
            return torch.float32

        dtype_map = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }
        return dtype_map.get(self._compute_precision, torch.float16)

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
        """Run inference returning standardized batched output.

        SigLIP can encode text, images, or both. For items with only text,
        returns text embeddings. For items with only images, returns image
        embeddings.

        Args:
            items: List of items to encode (with text and/or images).
            output_types: Which outputs to return (only "dense" supported).
            instruction: Optional instruction (not used by SigLIP).
            is_query: Whether items are queries.
            prepared_items: Not used by this adapter.

        Returns:
            EncodeOutput with dense embeddings.
        """
        self._check_loaded()
        if self._processor is None:
            raise RuntimeError(ERR_NOT_LOADED)

        self._validate_output_types(output_types)

        # Encode each item individually and stack into batch
        import numpy as np

        embeddings_list = []
        for item in items:
            embedding = self._encode_single_item(item)
            embeddings_list.append(embedding)

        # Stack into batched array [batch, dim]
        dense_batch = np.stack(embeddings_list, axis=0)

        return EncodeOutput(
            dense=dense_batch,
            batch_size=len(items),
            is_query=is_query,
            dense_dim=self._dense_dim,
        )

    def _encode_single_item(self, item: Item) -> Any:
        """Encode a single item (text, image, or both).

        Returns:
            Numpy array of shape [dense_dim].
        """
        has_text = item.text is not None
        images = item.images
        has_images = images is not None and len(images) > 0

        if not has_text and not has_images:
            raise ValueError(_ERR_NO_INPUT)

        # Determine what to encode
        if has_images:
            # Image encoding (or image+text where image takes precedence)
            pil_images = self._load_images(item)
            return self._encode_images(pil_images)
        # Text-only encoding (text is guaranteed non-None if no images)
        return self._encode_text(item.text)  # type: ignore

    def _load_images(self, item: Item) -> list[Image.Image]:
        """Load images from item into PIL Images.

        Args:
            item: Item with images field.

        Returns:
            List of PIL Images.
        """
        from PIL import Image

        pil_images = []
        for img_input in item.images or []:
            # img_input is ImageInput TypedDict with data (bytes) and optional format
            img_bytes = img_input["data"]
            pil_img = Image.open(io.BytesIO(img_bytes))
            # Convert to RGB if necessary
            if pil_img.mode != "RGB":
                pil_img = pil_img.convert("RGB")
            pil_images.append(pil_img)

        return pil_images

    def _encode_images(self, images: list[Image.Image]) -> Any:
        """Encode images into embeddings.

        Args:
            images: List of PIL Images.

        Returns:
            Numpy array of shape [dense_dim] (averaged if multiple images).
        """
        assert self._model is not None
        assert self._processor is not None

        from torch.nn import functional

        # Process images
        inputs = self._processor(images=images, return_tensors="pt")
        inputs = {k: v.to(self._device) for k, v in inputs.items()}

        with torch.inference_mode():
            image_features = self._model.get_image_features(**inputs)

            # L2 normalize if configured
            if self._normalize:
                image_features = functional.normalize(image_features, p=2, dim=-1)

        # If multiple images, average the embeddings
        if len(images) > 1:
            image_features = image_features.mean(dim=0, keepdim=True)

        return image_features[0].float().cpu().numpy()

    def _encode_text(self, text: str) -> Any:
        """Encode text into embeddings.

        Args:
            text: Text string to encode.

        Returns:
            Numpy array of shape [dense_dim].
        """
        assert self._model is not None
        assert self._processor is not None

        from torch.nn import functional

        # Process text - use max_length padding to match MTEB behavior
        # SigLIP text embeddings depend on sequence length, so consistent padding is required
        inputs = self._processor(text=[text], return_tensors="pt", padding="max_length", truncation=True)
        inputs = {k: v.to(self._device) for k, v in inputs.items()}

        with torch.inference_mode():
            text_features = self._model.get_text_features(**inputs)

            # L2 normalize if configured
            if self._normalize:
                text_features = functional.normalize(text_features, p=2, dim=-1)

        return text_features[0].float().cpu().numpy()

    def _validate_output_types(self, output_types: list[str]) -> None:
        """Validate that output types are supported."""
        unsupported = set(output_types) - {"dense"}
        if unsupported:
            msg = f"Unsupported output types: {unsupported}. SigLIP only supports 'dense'."
            raise ValueError(msg)

    def get_preprocessor(self) -> Any | None:
        """Return an ImagePreprocessor for CPU/GPU overlap.

        Returns:
            ImagePreprocessor wrapping the SiglipProcessor, or None if not loaded.
        """
        if self._processor is None:
            return None

        from sie_server.core.preprocessor import ImagePreprocessor

        return ImagePreprocessor(self._processor, self._model_name_or_path)
