"""Image preprocessor for standard image models (CLIP, SigLIP).

This module contains the basic image preprocessor for contrastive
image-text models like CLIP and SigLIP that take single images.
"""

from __future__ import annotations

import io
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from sie_server.core.prepared import ImagePayload, PreparedBatch, PreparedItem

if TYPE_CHECKING:
    import torch
    from PIL import Image as PILImageType

    from sie_server.config.model import ModelConfig
    from sie_server.types.inputs import Item


class ImagePreprocessor:
    """Preprocessor for image processing.

    Handles PIL Image loading from bytes and processor transformation.
    Thread-safe: PIL and processors handle concurrent calls.
    """

    def __init__(
        self,
        processor: Any,  # SiglipProcessor, CLIPProcessor, etc.
        model_name: str,
    ) -> None:
        """Initialize with an image processor.

        Args:
            processor: HuggingFace processor with image_processor.
            model_name: Model name for logging.
        """
        self._processor = processor
        self._model_name = model_name

    @property
    def modality(self) -> str:
        """Return 'image'."""
        return "image"

    def prepare(
        self,
        items: list[Item],
        *,
        config: ModelConfig,
        is_query: bool = False,
        instruction: str | None = None,
        task: str | None = None,
    ) -> PreparedBatch[ImagePayload]:
        """Process images from items.

        Each item may have multiple images. For now, we take the first image
        per item (matching current adapter behavior).

        Args:
            items: Items with images field.
            config: Model config (unused for images currently).
            is_query: Whether items are queries (unused for standard images).
            instruction: Optional instruction (unused for standard image preprocessing).
            task: Optional task token (unused for standard image preprocessing).

        Returns:
            PreparedBatch with ImagePayload items.
        """
        from PIL import Image as PILImage

        prepared_items: list[PreparedItem[ImagePayload]] = []
        total_cost = 0

        for i, item in enumerate(items):
            if not item.images:
                # Skip items without images (they may be text-only)
                continue

            # Load first image from bytes
            img_input = item.images[0]
            pil_img = PILImage.open(io.BytesIO(img_input["data"]))
            original_size = pil_img.size

            # Convert to RGB if needed
            if pil_img.mode != "RGB":
                pil_img = pil_img.convert("RGB")

            # Process through HuggingFace processor
            processed = self._processor(images=pil_img, return_tensors="pt")
            pixel_values = processed["pixel_values"].squeeze(0)  # Remove batch dim

            payload = ImagePayload(
                pixel_values=pixel_values,
                original_size=original_size,
            )
            # Cost = 1 per image (fixed dimensions after resize)
            prepared_items.append(PreparedItem(payload=payload, cost=1, original_index=i))
            total_cost += 1

        return PreparedBatch(
            items=prepared_items,
            total_cost=total_cost,
            modality="image",
        )

    def collate(
        self,
        prepared: list[PreparedItem[ImagePayload]],
        *,
        device: str,
    ) -> dict[str, Any]:
        """Collate image items into batched tensor.

        Args:
            prepared: List of prepared image items.
            device: Target device.

        Returns:
            Dict with 'pixel_values' tensor of shape [B, C, H, W].
        """
        import torch

        if not prepared:
            return {"pixel_values": torch.tensor([])}

        # Stack pixel values into batch
        pixel_values = torch.stack([p.payload.pixel_values for p in prepared])

        return {"pixel_values": pixel_values.to(device)}


class OpenCLIPImagePreprocessor:
    """Preprocessor for ``open_clip``-native models.

    Adapts the ``val_preproc`` callable returned by
    ``open_clip.create_model_and_transforms`` (a ``torchvision.transforms``
    pipeline mapping ``PIL.Image -> Tensor[C, H, W]``) to the
    ``Preprocessor`` protocol shared by adapters in this codebase. This lets
    the framework run PIL decoding + resize on a CPU executor thread in
    parallel with the previous batch's GPU forward pass, matching the
    CPU/GPU-overlap optimization that ``ImagePreprocessor`` provides for
    HuggingFace processors.

    Thread-safety: assumes ``val_preproc`` is stateless and reentrant. The
    pipeline returned by ``open_clip.create_model_and_transforms`` (resize +
    center crop + ``ToTensor`` + ``Normalize``) satisfies this. Custom
    callables that share mutable state (e.g. transforms with non-thread-local
    RNG) are the caller's responsibility.
    """

    def __init__(
        self,
        val_preproc: Callable[[PILImageType.Image], torch.Tensor],
        model_name: str,
    ) -> None:
        """Initialize with an open_clip val_preproc callable.

        Args:
            val_preproc: The ``val_preproc`` callable returned by
                ``open_clip.create_model_and_transforms`` (a torchvision
                Compose). Maps ``PIL.Image -> Tensor[C, H, W]`` already
                normalized for the model.
            model_name: Model name for logging.
        """
        self._val_preproc = val_preproc
        self._model_name = model_name

    @property
    def modality(self) -> str:
        """Return 'image'."""
        return "image"

    def prepare(
        self,
        items: list[Item],
        *,
        config: ModelConfig,
        is_query: bool = False,
        instruction: str | None = None,
        task: str | None = None,
    ) -> PreparedBatch[ImagePayload]:
        """Process images from items via open_clip's val_preproc.

        Each item may have multiple images. As with ``ImagePreprocessor``, we
        take the first image per item to match the encode-side behavior.

        Args:
            items: Items with images field.
            config: Model config (unused for images currently).
            is_query: Whether items are queries (unused for standard images).
            instruction: Optional instruction (unused for standard image preprocessing).
            task: Optional task token (unused for standard image preprocessing).

        Returns:
            PreparedBatch with ImagePayload items.
        """
        from PIL import Image as PILImage

        prepared_items: list[PreparedItem[ImagePayload]] = []
        total_cost = 0

        for i, item in enumerate(items):
            if not item.images:
                # Skip items without images (they may be text-only)
                continue

            # Load first image from bytes
            img_input = item.images[0]
            pil_img = PILImage.open(io.BytesIO(img_input["data"]))
            original_size = pil_img.size

            # Convert to RGB if needed
            if pil_img.mode != "RGB":
                pil_img = pil_img.convert("RGB")

            # open_clip val_preproc returns a [C, H, W] tensor (no batch dim)
            pixel_values = self._val_preproc(pil_img)

            payload = ImagePayload(
                pixel_values=pixel_values,
                original_size=original_size,
            )
            # Cost = 1 per image (fixed dimensions after resize)
            prepared_items.append(PreparedItem(payload=payload, cost=1, original_index=i))
            total_cost += 1

        return PreparedBatch(
            items=prepared_items,
            total_cost=total_cost,
            modality="image",
        )

    def collate(
        self,
        prepared: list[PreparedItem[ImagePayload]],
        *,
        device: str,
    ) -> dict[str, Any]:
        """Collate image items into batched tensor.

        Args:
            prepared: List of prepared image items.
            device: Target device.

        Returns:
            Dict with 'pixel_values' tensor of shape [B, C, H, W].
        """
        import torch

        if not prepared:
            return {"pixel_values": torch.tensor([])}

        # Stack pixel values into batch
        pixel_values = torch.stack([p.payload.pixel_values for p in prepared])

        return {"pixel_values": pixel_values.to(device)}
