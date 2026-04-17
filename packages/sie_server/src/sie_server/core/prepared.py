"""Prepared item types for modality-agnostic batching.

This module defines the core abstractions for preprocessing items before
batching and inference. These types generalize TokenizedItem to support
text, images, audio, and mixed-modality inputs.

See roadmap.md Project 10.4 Phase 2 for design details.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(slots=True)
class Payload:
    """Base class for modality-specific payloads.

    Subclasses carry preprocessed data ready for batching.
    """


@dataclass(slots=True)
class TextPayload(Payload):
    """Tokenized text ready for batching.

    Equivalent to the existing TokenizedItem but as a payload type.
    """

    input_ids: list[int]
    attention_mask: list[int]

    @property
    def token_count(self) -> int:
        """Return number of tokens."""
        return len(self.input_ids)


@dataclass(slots=True)
class ImagePayload(Payload):
    """Preprocessed image ready for batching.

    Contains pixel values after processor transformation.
    For SigLIP/CLIP: [C, H, W] tensor after resize/normalize.
    """

    pixel_values: Any  # torch.Tensor, but avoid import at module level
    original_size: tuple[int, int]  # (width, height) for debugging


@dataclass(slots=True)
class AudioPayload(Payload):
    """Preprocessed audio ready for batching (future).

    Contains waveform data after resampling/normalization.
    """

    waveform: Any  # torch.Tensor
    sample_rate: int
    duration_s: float


@dataclass(slots=True)
class PreparedItem[T: Payload]:
    """A prepared item ready for batching.

    Generic over the payload type (TextPayload, ImagePayload, etc.).
    The cost field is used by BatchFormer to form optimal batches.

    Attributes:
        payload: Modality-specific preprocessed data.
        cost: Batching budget consumed (tokens for text, 1 for images).
        original_index: Position in original request for result reordering.
    """

    payload: T
    cost: int
    original_index: int


@dataclass(slots=True)
class PreparedBatch[T: Payload]:
    """Batch of prepared items with total cost.

    Attributes:
        items: List of prepared items in this batch.
        total_cost: Sum of costs for batching decisions.
        modality: Primary modality: "text", "image", "audio", "mixed".
    """

    items: list[PreparedItem[T]]
    total_cost: int
    modality: str = "text"

    @property
    def size(self) -> int:
        """Return number of items in batch."""
        return len(self.items)

    def sorted_by_cost(self) -> PreparedBatch[T]:
        """Return new batch with items sorted by cost (ascending).

        Sorting reduces padding waste when collating to tensors.
        """
        sorted_items = sorted(self.items, key=lambda x: x.cost)
        return PreparedBatch(
            items=sorted_items,
            total_cost=self.total_cost,
            modality=self.modality,
        )


# Type aliases for common prepared types
TextPreparedItem = PreparedItem[TextPayload]
ImagePreparedItem = PreparedItem[ImagePayload]
AudioPreparedItem = PreparedItem[AudioPayload]

# Union type for any prepared item
AnyPreparedItem = PreparedItem[TextPayload] | PreparedItem[ImagePayload] | PreparedItem[AudioPayload]


@dataclass(slots=True)
class ExtractPreparedItem:
    """Prepared item for extract operations.

    Satisfies HasCost protocol for BatchFormer.
    Cost is based on character count (no tokenization needed - GLiNER does its own).

    Attributes:
        cost: Character count of the text (approximate batching budget).
        original_index: Position in original request for result reordering.
    """

    cost: int
    original_index: int


@dataclass(slots=True)
class ScorePreparedItem:
    """Prepared item for score (reranking) operations.

    Satisfies HasCost protocol for BatchFormer.
    Cost is based on combined character count of query + document.
    Cross-encoder rerankers process (query, doc) pairs together.

    Attributes:
        cost: Character count of query + doc (approximate batching budget).
        original_index: Index of this doc in the request's items list.
    """

    cost: int
    original_index: int


@dataclass
class MixedPayload(Payload):
    """Mixed-modality payload containing multiple modality payloads.

    Used when an item has both text and images (e.g., ColPali with OCR).
    """

    text: TextPayload | None = None
    image: ImagePayload | None = None
    audio: AudioPayload | None = None

    @property
    def modalities(self) -> list[str]:
        """Return list of present modalities."""
        result = []
        if self.text is not None:
            result.append("text")
        if self.image is not None:
            result.append("image")
        if self.audio is not None:
            result.append("audio")
        return result


@dataclass(slots=True)
class NemoColEmbedPayload(Payload):
    """Preprocessed NemoColEmbed input ready for batching.

    NemoColEmbed uses dynamic image tiling (1-6 tiles per image based on aspect ratio)
    combined with tokenized prompts containing <IMG_CONTEXT> placeholder tokens.

    The model's forward() expects:
    - pixel_values: [num_tiles, C, H, W] tensor for vision encoder
    - input_ids: [seq_len] tokenized prompt with image placeholder tokens
    - attention_mask: [seq_len] attention mask

    Attributes:
        pixel_values: Stacked pixel tensors for all tiles [num_tiles, 3, 448, 448].
        input_ids: Tokenized prompt with <IMG_CONTEXT> tokens [seq_len].
        attention_mask: Attention mask for the prompt [seq_len].
        num_tiles: Number of tiles (1-6) for cost calculation.
        original_size: Original image size (width, height) for debugging.
    """

    pixel_values: Any  # torch.Tensor [num_tiles, 3, 448, 448]
    input_ids: Any  # torch.Tensor [seq_len]
    attention_mask: Any  # torch.Tensor [seq_len]
    num_tiles: int
    original_size: tuple[int, int]


# Type alias for NemoColEmbed prepared item
NemoColEmbedPreparedItem = PreparedItem[NemoColEmbedPayload]


@dataclass(slots=True)
class Florence2Payload(Payload):
    """Preprocessed Florence-2 input ready for extraction.

    Florence-2 uses a DaViT vision encoder + BART decoder for document understanding.
    The processor transforms images and text prompts into model inputs.

    Attributes:
        pixel_values: Preprocessed image tensor [C, H, W] (fp16/bf16).
        input_ids: Tokenized task prompt [seq_len].
        attention_mask: Attention mask for the prompt [seq_len].
        original_size: Original image size (width, height) for bbox normalization.
    """

    pixel_values: Any  # torch.Tensor [C, H, W]
    input_ids: Any  # torch.Tensor [seq_len]
    attention_mask: Any  # torch.Tensor [seq_len]
    original_size: tuple[int, int]


# Type alias for Florence-2 prepared item
Florence2PreparedItem = PreparedItem[Florence2Payload]


@dataclass(slots=True)
class DonutPayload(Payload):
    """Preprocessed Donut input ready for extraction.

    Donut uses a Swin vision encoder + BART decoder for document understanding.
    The processor transforms images and decoder prompts into model inputs.

    Attributes:
        pixel_values: Preprocessed image tensor [C, H, W] (fp16/bf16).
        decoder_input_ids: Tokenized decoder prompt [seq_len].
        original_size: Original image size (width, height) for debugging.
    """

    pixel_values: Any  # torch.Tensor [C, H, W]
    decoder_input_ids: Any  # torch.Tensor [seq_len]
    original_size: tuple[int, int]


# Type alias for Donut prepared item
DonutPreparedItem = PreparedItem[DonutPayload]


@dataclass(slots=True)
class LightOnOCRPayload(Payload):
    """Preprocessed LightOnOCR input ready for extraction.

    LightOnOCR-2-1B uses a Pixtral vision encoder + Qwen3 text decoder
    (Mistral3 architecture). The processor applies a chat template and
    processes images into model inputs.

    Attributes:
        pixel_values: Preprocessed image tensor [C, H, W].
        input_ids: Tokenized chat prompt [seq_len].
        attention_mask: Attention mask for the prompt [seq_len].
        image_sizes: Image dimensions tensor [2] required by generate().
        original_size: Original image size (width, height).
    """

    pixel_values: Any  # torch.Tensor [C, H, W]
    input_ids: Any  # torch.Tensor [seq_len]
    attention_mask: Any  # torch.Tensor [seq_len]
    image_sizes: Any  # torch.Tensor [2]
    original_size: tuple[int, int]


# Type alias for LightOnOCR prepared item
LightOnOCRPreparedItem = PreparedItem[LightOnOCRPayload]


@dataclass(slots=True)
class DetectionPayload(Payload):
    """Preprocessed detection model input ready for inference.

    Used by GroundingDINO and OWL-v2 adapters for open-vocabulary detection.
    The processor transforms images into model inputs (resize, normalize).

    Attributes:
        pixel_values: Preprocessed image tensor [C, H, W].
        original_size: Original image size (width, height) for bbox denormalization.
    """

    pixel_values: Any  # torch.Tensor [C, H, W]
    original_size: tuple[int, int]


# Type alias for detection prepared item
DetectionPreparedItem = PreparedItem[DetectionPayload]


def make_text_item(
    input_ids: list[int],
    original_index: int = 0,
    *,
    attention_mask: list[int] | None = None,
) -> PreparedItem[TextPayload]:
    """Create a PreparedItem[TextPayload] for testing.

    Convenience factory that creates a text prepared item with sensible defaults.
    The cost is automatically set to the number of tokens (len(input_ids)).

    Args:
        input_ids: Token IDs.
        original_index: Position in original request (default 0).
        attention_mask: Optional attention mask. Defaults to all 1s.

    Returns:
        PreparedItem with TextPayload.

    Example:
        item = make_text_item([1, 2, 3])  # 3 tokens, index 0
        item = make_text_item([1, 2, 3, 4, 5], original_index=2)
    """
    if attention_mask is None:
        attention_mask = [1] * len(input_ids)
    return PreparedItem(
        payload=TextPayload(input_ids=input_ids, attention_mask=attention_mask),
        cost=len(input_ids),
        original_index=original_index,
    )
