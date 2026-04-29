"""Vision preprocessors for document understanding and detection models.

This module contains specialized vision preprocessors for:
- NemoColEmbedPreprocessor: Visual document retrieval with dynamic tiling
- Florence2Preprocessor: Florence-2 document understanding
- DonutPreprocessor: Donut document understanding
- DetectionPreprocessor: Object detection (GroundingDINO, OWL-v2)
"""

from __future__ import annotations

import io
import logging
from typing import TYPE_CHECKING, Any

from sie_server.core.prepared import (
    DetectionPayload,
    DonutPayload,
    Florence2Payload,
    GlmOcrPayload,
    LightOnOCRPayload,
    NemoColEmbedPayload,
    PaddleOCRVLPayload,
    PreparedBatch,
    PreparedItem,
)
from sie_server.core.preprocessor.base import get_image_executor

if TYPE_CHECKING:
    from sie_server.config.model import ModelConfig
    from sie_server.types.inputs import Item

logger = logging.getLogger(__name__)

# Constants for NemoColEmbed preprocessing (from NVIDIA's code)
_SIGLIP_MEAN = (0.5, 0.5, 0.5)
_SIGLIP_STD = (0.5, 0.5, 0.5)


def _find_closest_aspect_ratio(
    aspect_ratio: float,
    target_ratios: list[tuple[int, int]],
    width: int,
    height: int,
    image_size: int,
) -> tuple[int, int]:
    """Find the best tiling ratio for an image.

    Considers both aspect ratio match and area coverage.
    Extracted from NVIDIA's modeling_llama_nemoretrievercolembed.py.
    """
    best_factor = float("-inf")
    best_ratio = (1, 1)
    area = width * height

    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        factor_based_on_area_n_ratio = min((ratio[0] * ratio[1] * image_size * image_size) / area, 0.6) * min(
            target_aspect_ratio / aspect_ratio, aspect_ratio / target_aspect_ratio
        )

        if factor_based_on_area_n_ratio > best_factor:
            best_factor = factor_based_on_area_n_ratio
            best_ratio = ratio

    return best_ratio


def _dynamic_preprocess(
    image: Any,  # PIL.Image
    min_num: int = 1,
    max_num: int = 6,
    image_size: int = 448,
    use_thumbnail: bool = False,
) -> list[Any]:
    """Split image into tiles based on aspect ratio.

    Extracted from NVIDIA's modeling_llama_nemoretrievercolembed.py.

    Args:
        image: PIL Image to process.
        min_num: Minimum number of tiles (default 1).
        max_num: Maximum number of tiles (default 6).
        image_size: Size of each tile (default 448).
        use_thumbnail: Whether to add thumbnail tile.

    Returns:
        List of PIL Image tiles (1-6 tiles, plus optional thumbnail).
    """
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # Calculate target ratios (all valid tile arrangements from min_num to max_num)
    target_ratios = sorted(
        {
            (i, j)
            for n in range(min_num, max_num + 1)
            for i in range(1, n + 1)
            for j in range(1, n + 1)
            if min_num <= i * j <= max_num
        },
        key=lambda x: x[0] * x[1],
    )

    # Find closest aspect ratio
    target_aspect_ratio = _find_closest_aspect_ratio(aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    # Calculate target dimensions
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # Resize and crop into tiles
    resized_img = image.resize((target_width, target_height))
    processed_images = []

    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size,
        )
        split_img = resized_img.crop(box)
        processed_images.append(split_img)

    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)

    return processed_images


class NemoColEmbedPreprocessor:
    """Preprocessor for NemoColEmbed visual document retrieval model.

    Handles the full preprocessing pipeline:
    1. Dynamic image tiling (1-6 tiles based on aspect ratio)
    2. Image transform (resize, normalize with SigLIP constants)
    3. Tokenization with template and <IMG_CONTEXT> placeholder tokens

    This replaces the model's internal DataLoader-based preprocessing,
    allowing SIE to use its own thread pool and batching infrastructure.

    Thread-safe: PIL, torchvision transforms, and tokenizers handle concurrent calls.
    """

    def __init__(
        self,
        tokenizer: Any,
        model_config: Any,
        model_name: str,
        *,
        image_size: int = 448,
        max_input_tiles: int = 6,
        use_thumbnail: bool = False,
        num_image_token: int = 256,
    ) -> None:
        """Initialize with tokenizer and model configuration.

        Args:
            tokenizer: HuggingFace tokenizer from the model.
            model_config: Model configuration with template info.
            model_name: Model name for logging.
            image_size: Size of each tile (default 448).
            max_input_tiles: Maximum tiles per image (default 6).
            use_thumbnail: Whether to add thumbnail tile (default False).
            num_image_token: Number of image tokens per tile (default 256).
        """
        self._tokenizer = tokenizer
        self._model_config = model_config
        self._model_name = model_name
        self._image_size = image_size
        self._max_input_tiles = max_input_tiles
        self._use_thumbnail = use_thumbnail
        self._num_image_token = num_image_token

        # Build image transform (matches NVIDIA's build_transform with siglip norm)
        self._transform = self._build_transform()

    def _build_transform(self) -> Any:
        """Build the image transform pipeline."""
        import torchvision.transforms as T
        from torchvision.transforms.functional import InterpolationMode

        return T.Compose(
            [
                T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
                T.Resize(
                    (self._image_size, self._image_size),
                    interpolation=InterpolationMode.BICUBIC,
                ),
                T.ToTensor(),
                T.Normalize(mean=_SIGLIP_MEAN, std=_SIGLIP_STD),
            ]
        )

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
    ) -> PreparedBatch[NemoColEmbedPayload]:
        """Process images for NemoColEmbed model.

        For each item:
        1. Load image from bytes
        2. Apply dynamic tiling
        3. Transform each tile
        4. Build prompt with <IMG_CONTEXT> tokens
        5. Tokenize

        Args:
            items: Items with images field.
            config: Model config (uses internal config instead).
            is_query: Whether items are queries (unused for NemoColEmbed images).
            instruction: Optional instruction (unused).
            task: Optional task token (unused).

        Returns:
            PreparedBatch with NemoColEmbedPayload items.
        """
        import torch
        from PIL import Image as PILImage

        prepared_items: list[PreparedItem[NemoColEmbedPayload]] = []
        total_cost = 0

        for i, item in enumerate(items):
            if not item.images:
                logger.warning("NemoColEmbedPreprocessor: item %d has no images", i)
                continue

            # Load image from bytes
            img_input = item.images[0]
            pil_img = PILImage.open(io.BytesIO(img_input["data"]))
            original_size = pil_img.size

            # Convert to RGB if needed
            if pil_img.mode != "RGB":
                pil_img = pil_img.convert("RGB")

            # Dynamic tiling
            image_tiles = _dynamic_preprocess(
                pil_img,
                image_size=self._image_size,
                max_num=self._max_input_tiles,
                use_thumbnail=self._use_thumbnail,
            )
            num_tiles = len(image_tiles)

            # Transform each tile and stack
            pixel_values_list = [self._transform(tile) for tile in image_tiles]
            pixel_values = torch.stack(pixel_values_list).to(dtype=torch.bfloat16)

            # Build prompt with image tokens
            prompt = self._build_prompt(num_tiles)

            # Tokenize
            encoded = self._tokenizer(
                prompt,
                truncation=True,
                max_length=10240,  # p_max_length from model
                return_tensors="pt",
            )

            payload = NemoColEmbedPayload(
                pixel_values=pixel_values,
                input_ids=encoded["input_ids"].squeeze(0),
                attention_mask=encoded["attention_mask"].squeeze(0),
                num_tiles=num_tiles,
                original_size=original_size,
            )

            # Cost = number of tiles (variable per image, affects GPU memory)
            prepared_items.append(PreparedItem(payload=payload, cost=num_tiles, original_index=i))
            total_cost += num_tiles

        return PreparedBatch(
            items=prepared_items,
            total_cost=total_cost,
            modality="image",
        )

    def _build_prompt(self, num_tiles: int) -> str:
        """Build prompt with image placeholder tokens.

        The prompt format matches NVIDIA's process_documents():
        "passage: <img><IMG_CONTEXT>*N</img> " where N = num_tiles * num_image_token

        Note: The model uses 'bidirectional-llama-retriever' template which
        has empty roles and separators, so the template is essentially a no-op.
        The trailing space comes from: prefix + ' ' + content where prefix='<image>'.

        Args:
            num_tiles: Number of image tiles.

        Returns:
            Formatted prompt string.
        """
        # Template tokens
        img_start = "<img>"
        img_end = "</img>"
        img_context = "<IMG_CONTEXT>"

        # Build image tokens placeholder
        num_context_tokens = num_tiles * self._num_image_token
        image_tokens = img_start + (img_context * num_context_tokens) + img_end

        # Prompt format matches: "passage: <image> " after template processing
        # with <image> replaced by the actual image tokens
        return f"passage: {image_tokens} "

    def collate(
        self,
        prepared: list[PreparedItem[NemoColEmbedPayload]],
        *,
        device: str,
    ) -> dict[str, Any]:
        """Collate NemoColEmbed items into batched tensors.

        Note: NemoColEmbed has variable tile counts per image, so pixel_values
        are concatenated (not stacked) into [total_tiles, C, H, W].

        Args:
            prepared: List of prepared items.
            device: Target device.

        Returns:
            Dict with 'pixel_values', 'input_ids', 'attention_mask' tensors.
        """
        import torch

        if not prepared:
            return {
                "pixel_values": torch.tensor([]),
                "input_ids": torch.tensor([]),
                "attention_mask": torch.tensor([]),
            }

        # Concatenate pixel values (variable tiles per item)
        all_pixel_values = torch.cat([p.payload.pixel_values for p in prepared], dim=0)

        # Pad input_ids and attention_mask to same length
        max_length = max(p.payload.input_ids.shape[0] for p in prepared)
        pad_token_id = self._tokenizer.pad_token_id or 0

        input_ids_batch = []
        attention_mask_batch = []

        for p in prepared:
            ids = p.payload.input_ids
            mask = p.payload.attention_mask
            padding_length = max_length - ids.shape[0]

            if padding_length > 0:
                ids = torch.cat(
                    [
                        torch.full((padding_length,), pad_token_id, dtype=ids.dtype),
                        ids,
                    ]
                )  # Left padding
                mask = torch.cat(
                    [
                        torch.zeros(padding_length, dtype=mask.dtype),
                        mask,
                    ]
                )

            input_ids_batch.append(ids)
            attention_mask_batch.append(mask)

        return {
            "pixel_values": all_pixel_values.to(device),
            "input_ids": torch.stack(input_ids_batch).to(device),
            "attention_mask": torch.stack(attention_mask_batch).to(device),
        }


class Florence2Preprocessor:
    """Preprocessor for Florence-2 document understanding model.

    Handles CPU-bound image preprocessing:
    1. Load image from bytes (PIL)
    2. Convert to RGB if needed
    3. Process through Florence-2 processor (resize, normalize)
    4. Tokenize task prompt

    This moves heavy image processing off the GPU thread, allowing
    overlap with inference on previous batches.

    Thread-safe: PIL and HuggingFace processors handle concurrent calls.
    """

    def __init__(
        self,
        processor: Any,  # Florence-2 processor
        model_name: str,
        *,
        default_task: str = "<OCR_WITH_REGION>",
    ) -> None:
        """Initialize with a Florence-2 processor.

        Args:
            processor: HuggingFace AutoProcessor for Florence-2.
            model_name: Model name for logging.
            default_task: Default task prompt token.
        """
        self._processor = processor
        self._model_name = model_name
        self._default_task = default_task

    @property
    def modality(self) -> str:
        """Return 'image'."""
        return "image"

    def _process_single_image(
        self,
        item: Item,
        index: int,
        prompt: str,
    ) -> PreparedItem[Florence2Payload] | None:
        """Process a single image item (thread-safe).

        Args:
            item: Item with images field.
            index: Original index for reordering.
            prompt: Task prompt string.

        Returns:
            PreparedItem or None if item has no images.
        """
        from PIL import Image as PILImage

        if not item.images:
            logger.warning("Florence2Preprocessor: item %d has no images", index)
            return None

        # Load image from bytes - PIL releases GIL during decode
        img_input = item.images[0]
        pil_img = PILImage.open(io.BytesIO(img_input["data"]))
        original_size = (pil_img.width, pil_img.height)

        # Convert to RGB if needed
        if pil_img.mode != "RGB":
            pil_img = pil_img.convert("RGB")

        # Process through Florence-2 processor (CPU-bound, releases GIL)
        inputs = self._processor(
            text=prompt,
            images=pil_img,
            return_tensors="pt",
        )

        payload = Florence2Payload(
            pixel_values=inputs["pixel_values"].squeeze(0),
            input_ids=inputs["input_ids"].squeeze(0),
            attention_mask=inputs["attention_mask"].squeeze(0),
            original_size=original_size,
        )

        return PreparedItem(payload=payload, cost=1, original_index=index)

    def prepare(
        self,
        items: list[Item],
        *,
        config: ModelConfig,
        is_query: bool = False,
        task: str | None = None,
        instruction: str | None = None,
    ) -> PreparedBatch[Florence2Payload]:
        """Preprocess images and prompts for Florence-2.

        Uses parallel processing for batches of 2+ images to utilize
        multiple CPU cores (PIL releases GIL during image operations).

        Args:
            items: Items with images field.
            config: Model config (unused - uses processor defaults).
            is_query: Whether items are queries (unused for Florence-2).
            task: Task prompt token (e.g., "<OCR_WITH_REGION>").
            instruction: Optional instruction to append to task.

        Returns:
            PreparedBatch with Florence2Payload items.
        """
        # Build prompt
        prompt = task or self._default_task
        if instruction:
            prompt = f"{prompt}{instruction}"

        # Single item: process directly (no thread pool overhead)
        if len(items) == 1:
            result = self._process_single_image(items[0], 0, prompt)
            if result is None:
                return PreparedBatch(items=[], total_cost=0, modality="image")
            return PreparedBatch(items=[result], total_cost=1, modality="image")

        # Multiple items: parallel processing
        executor = get_image_executor()
        futures = [executor.submit(self._process_single_image, item, i, prompt) for i, item in enumerate(items)]

        # Collect results in order
        prepared_items: list[PreparedItem[Florence2Payload]] = []
        for future in futures:
            result = future.result()
            if result is not None:
                prepared_items.append(result)

        return PreparedBatch(
            items=prepared_items,
            total_cost=len(prepared_items),
            modality="image",
        )

    def collate(
        self,
        prepared: list[PreparedItem[Florence2Payload]],
        *,
        device: str,
        dtype: Any = None,
    ) -> dict[str, Any]:
        """Collate Florence-2 items into batched tensors.

        Args:
            prepared: List of prepared items.
            device: Target device.
            dtype: Optional dtype for pixel_values (e.g., torch.float16).

        Returns:
            Dict with 'pixel_values', 'input_ids', 'attention_mask' tensors.
        """
        import torch

        if not prepared:
            return {
                "pixel_values": torch.tensor([]),
                "input_ids": torch.tensor([]),
                "attention_mask": torch.tensor([]),
            }

        # Stack pixel values (all same size after processor)
        pixel_values = torch.stack([p.payload.pixel_values for p in prepared])
        if dtype is not None:
            pixel_values = pixel_values.to(dtype=dtype)

        # Pad input_ids and attention_mask
        max_length = max(p.payload.input_ids.shape[0] for p in prepared)

        input_ids_batch = []
        attention_mask_batch = []

        for p in prepared:
            ids = p.payload.input_ids
            mask = p.payload.attention_mask
            padding_length = max_length - ids.shape[0]

            if padding_length > 0:
                # Right padding for decoder-style models
                ids = torch.cat([ids, torch.zeros(padding_length, dtype=ids.dtype)])
                mask = torch.cat([mask, torch.zeros(padding_length, dtype=mask.dtype)])

            input_ids_batch.append(ids)
            attention_mask_batch.append(mask)

        return {
            "pixel_values": pixel_values.to(device),
            "input_ids": torch.stack(input_ids_batch).to(device),
            "attention_mask": torch.stack(attention_mask_batch).to(device),
        }


class DonutPreprocessor:
    """Preprocessor for Donut document understanding model.

    Handles CPU-bound image preprocessing:
    1. Load image from bytes (PIL)
    2. Convert to RGB if needed
    3. Process through Donut processor (resize, normalize)
    4. Tokenize decoder prompt

    This moves heavy image processing off the GPU thread, allowing
    overlap with inference on previous batches.

    Thread-safe: PIL and HuggingFace processors handle concurrent calls.
    """

    def __init__(
        self,
        processor: Any,  # DonutProcessor
        model_name: str,
        *,
        default_task: str = "<s_cord-v2>",
    ) -> None:
        """Initialize with a Donut processor.

        Args:
            processor: HuggingFace DonutProcessor.
            model_name: Model name for logging.
            default_task: Default task prompt token.
        """
        self._processor = processor
        self._model_name = model_name
        self._default_task = default_task

    @property
    def modality(self) -> str:
        """Return 'image'."""
        return "image"

    def _process_single_image(
        self,
        item: Item,
        index: int,
        prompt: str,
    ) -> PreparedItem[DonutPayload] | None:
        """Process a single image item (thread-safe).

        Args:
            item: Item with images field.
            index: Original index for reordering.
            prompt: Decoder prompt string.

        Returns:
            PreparedItem or None if item has no images.
        """
        from PIL import Image as PILImage

        if not item.images:
            logger.warning("DonutPreprocessor: item %d has no images", index)
            return None

        # Load image from bytes - PIL releases GIL during decode
        img_input = item.images[0]
        pil_img = PILImage.open(io.BytesIO(img_input["data"]))
        original_size = (pil_img.width, pil_img.height)

        # Convert to RGB if needed
        if pil_img.mode != "RGB":
            pil_img = pil_img.convert("RGB")

        # Process image through Donut processor (CPU-bound, releases GIL)
        pixel_values = self._processor(pil_img, return_tensors="pt").pixel_values.squeeze(0)

        # Tokenize decoder input
        decoder_input_ids = self._processor.tokenizer(
            prompt,
            add_special_tokens=False,
            return_tensors="pt",
        ).input_ids.squeeze(0)

        payload = DonutPayload(
            pixel_values=pixel_values,
            decoder_input_ids=decoder_input_ids,
            original_size=original_size,
        )

        return PreparedItem(payload=payload, cost=1, original_index=index)

    def prepare(
        self,
        items: list[Item],
        *,
        config: ModelConfig,
        is_query: bool = False,
        task: str | None = None,
        instruction: str | None = None,
    ) -> PreparedBatch[DonutPayload]:
        """Preprocess images and prompts for Donut.

        Uses parallel processing for batches of 2+ images to utilize
        multiple CPU cores (PIL releases GIL during image operations).

        Args:
            items: Items with images field.
            config: Model config (unused - uses processor defaults).
            is_query: Whether items are queries (unused for Donut).
            task: Task prompt token (e.g., "<s_cord-v2>").
            instruction: Optional question for DocVQA.

        Returns:
            PreparedBatch with DonutPayload items.
        """
        # Build prompt
        task_token = task or self._default_task
        if task_token == "<s_docvqa>" and instruction:  # noqa: S105 — model task token, not a password
            # DocVQA format
            prompt = f"{task_token}<s_question>{instruction}</s_question><s_answer>"
        else:
            prompt = task_token

        # Single item: process directly (no thread pool overhead)
        if len(items) == 1:
            result = self._process_single_image(items[0], 0, prompt)
            if result is None:
                return PreparedBatch(items=[], total_cost=0, modality="image")
            return PreparedBatch(items=[result], total_cost=1, modality="image")

        # Multiple items: parallel processing
        executor = get_image_executor()
        futures = [executor.submit(self._process_single_image, item, i, prompt) for i, item in enumerate(items)]

        # Collect results in order
        prepared_items: list[PreparedItem[DonutPayload]] = []
        for future in futures:
            result = future.result()
            if result is not None:
                prepared_items.append(result)

        return PreparedBatch(
            items=prepared_items,
            total_cost=len(prepared_items),
            modality="image",
        )

    def collate(
        self,
        prepared: list[PreparedItem[DonutPayload]],
        *,
        device: str,
        dtype: Any = None,
    ) -> dict[str, Any]:
        """Collate Donut items into batched tensors.

        Args:
            prepared: List of prepared items.
            device: Target device.
            dtype: Optional dtype for pixel_values (e.g., torch.float16).

        Returns:
            Dict with 'pixel_values', 'decoder_input_ids' tensors.
        """
        import torch

        if not prepared:
            return {
                "pixel_values": torch.tensor([]),
                "decoder_input_ids": torch.tensor([]),
            }

        # Stack pixel values (all same size after processor)
        pixel_values = torch.stack([p.payload.pixel_values for p in prepared])
        if dtype is not None:
            pixel_values = pixel_values.to(dtype=dtype)

        # Pad decoder_input_ids (typically same length for same task)
        max_length = max(p.payload.decoder_input_ids.shape[0] for p in prepared)
        pad_token_id = self._processor.tokenizer.pad_token_id or 0

        decoder_ids_batch = []
        for p in prepared:
            ids = p.payload.decoder_input_ids
            padding_length = max_length - ids.shape[0]

            if padding_length > 0:
                ids = torch.cat([ids, torch.full((padding_length,), pad_token_id, dtype=ids.dtype)])

            decoder_ids_batch.append(ids)

        return {
            "pixel_values": pixel_values.to(device),
            "decoder_input_ids": torch.stack(decoder_ids_batch).to(device),
        }


class LightOnOCRPreprocessor:
    """Preprocessor for LightOnOCR document OCR model.

    Handles CPU-bound image preprocessing:
    1. Load image from bytes (PIL)
    2. Convert to RGB if needed
    3. Build chat messages with system prompt and image placeholder
    4. Apply chat template and process through PixtralProcessor
    5. Extract image_sizes from processor output

    This moves heavy image processing off the GPU thread, allowing
    overlap with inference on previous batches.

    Thread-safe: PIL and HuggingFace processors handle concurrent calls.
    """

    def __init__(
        self,
        processor: Any,
        model_name: str,
        *,
        system_prompt: str = "You are an OCR engine. Return the markdown representation of the document.",
    ) -> None:
        """Initialize with a LightOnOCR processor.

        Args:
            processor: HuggingFace AutoProcessor (PixtralProcessor).
            model_name: Model name for logging.
            system_prompt: System prompt for the chat template.
        """
        self._processor = processor
        self._model_name = model_name
        self._system_prompt = system_prompt

    @property
    def modality(self) -> str:
        """Return 'image'."""
        return "image"

    def _build_messages(self, instruction: str | None = None) -> list[dict[str, Any]]:
        """Build chat messages for the model.

        Args:
            instruction: Optional instruction to append to user content.

        Returns:
            List of message dicts with system and user roles.
        """
        user_content: list[dict[str, str]] = [{"type": "image"}]
        if instruction:
            user_content.append({"type": "text", "text": instruction})

        return [
            {"role": "system", "content": self._system_prompt},
            {"role": "user", "content": user_content},
        ]

    def _process_single_image(
        self,
        item: Item,
        index: int,
        text: str,
    ) -> PreparedItem[LightOnOCRPayload] | None:
        """Process a single image item (thread-safe).

        Args:
            item: Item with images field.
            index: Original index for reordering.
            text: Chat template text for the processor.

        Returns:
            PreparedItem or None if item has no images.
        """
        from PIL import Image as PILImage

        if not item.images:
            logger.warning("LightOnOCRPreprocessor: item %d has no images", index)
            return None

        img_input = item.images[0]
        pil_img = PILImage.open(io.BytesIO(img_input["data"]))
        original_size = (pil_img.width, pil_img.height)

        if pil_img.mode != "RGB":
            pil_img = pil_img.convert("RGB")

        inputs = self._processor(
            text=text,
            images=[pil_img],
            return_tensors="pt",
        )

        payload = LightOnOCRPayload(
            pixel_values=inputs["pixel_values"].squeeze(0),
            input_ids=inputs["input_ids"].squeeze(0),
            attention_mask=inputs["attention_mask"].squeeze(0),
            image_sizes=inputs["image_sizes"].squeeze(0),
            original_size=original_size,
        )

        return PreparedItem(payload=payload, cost=1, original_index=index)

    def prepare(
        self,
        items: list[Item],
        *,
        config: ModelConfig,
        is_query: bool = False,
        instruction: str | None = None,
        task: str | None = None,
    ) -> PreparedBatch[LightOnOCRPayload]:
        """Preprocess images for LightOnOCR.

        Uses parallel processing for batches of 2+ images to utilize
        multiple CPU cores (PIL releases GIL during image operations).

        Args:
            items: Items with images field.
            config: Model config (unused - uses processor defaults).
            is_query: Whether items are queries (unused for LightOnOCR).
            task: Task token (unused for LightOnOCR).
            instruction: Optional instruction to append to user message.

        Returns:
            PreparedBatch with LightOnOCRPayload items.
        """
        messages = self._build_messages(instruction)
        text = self._processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False,
        )

        if len(items) == 1:
            result = self._process_single_image(items[0], 0, text)
            if result is None:
                return PreparedBatch(items=[], total_cost=0, modality="image")
            return PreparedBatch(items=[result], total_cost=1, modality="image")

        executor = get_image_executor()
        futures = [executor.submit(self._process_single_image, item, i, text) for i, item in enumerate(items)]

        prepared_items: list[PreparedItem[LightOnOCRPayload]] = []
        for future in futures:
            result = future.result()
            if result is not None:
                prepared_items.append(result)

        return PreparedBatch(
            items=prepared_items,
            total_cost=len(prepared_items),
            modality="image",
        )

    def collate(
        self,
        prepared: list[PreparedItem[LightOnOCRPayload]],
        *,
        device: str,
        dtype: Any = None,
    ) -> dict[str, Any]:
        """Collate LightOnOCR items into batched tensors.

        Args:
            prepared: List of prepared items.
            device: Target device.
            dtype: Optional dtype for pixel_values.

        Returns:
            Dict with input tensors for the model.
        """
        import torch

        if not prepared:
            return {
                "pixel_values": torch.tensor([]),
                "input_ids": torch.tensor([]),
                "attention_mask": torch.tensor([]),
                "image_sizes": torch.tensor([]),
            }

        # LightOnOCR processes one image at a time (variable-size inputs)
        p = prepared[0]
        pixel_values = p.payload.pixel_values.unsqueeze(0)
        if dtype is not None:
            pixel_values = pixel_values.to(dtype=dtype)

        return {
            "pixel_values": pixel_values.to(device),
            "input_ids": p.payload.input_ids.unsqueeze(0).to(device),
            "attention_mask": p.payload.attention_mask.unsqueeze(0).to(device),
            "image_sizes": p.payload.image_sizes.unsqueeze(0).to(device),
        }


class GlmOcrPreprocessor:
    """Preprocessor for GLM-OCR document OCR model.

    Handles CPU-bound image preprocessing:
    1. Load image from bytes (PIL)
    2. Convert to RGB if needed
    3. Build chat messages with user text and embedded PIL image
    4. Apply chat template (tokenize=True, return_dict=True) to produce
       pixel_values, input_ids, and attention_mask in a single call.

    GLM-OCR uses a single-turn user message (no system prompt) and supports
    embedding a PIL image directly in the chat template via
    `{"type": "image", "image": pil_img}` (transformers 5.x).

    This moves heavy image processing off the GPU thread, allowing
    overlap with inference on previous batches.

    Thread-safe: PIL and HuggingFace processors handle concurrent calls.
    """

    def __init__(
        self,
        processor: Any,
        model_name: str,
        *,
        user_text: str = "Text Recognition:",
    ) -> None:
        """Initialize with a GLM-OCR processor.

        Args:
            processor: HuggingFace AutoProcessor for GLM-OCR.
            model_name: Model name for logging.
            user_text: Default user text prompt appended to the image.
        """
        self._processor = processor
        self._model_name = model_name
        self._user_text = user_text

    @property
    def modality(self) -> str:
        """Return 'image'."""
        return "image"

    def _process_single_image(
        self,
        item: Item,
        index: int,
        instruction: str | None = None,
    ) -> PreparedItem[GlmOcrPayload] | None:
        """Process a single image item (thread-safe).

        Args:
            item: Item with images field.
            index: Original index for reordering.
            instruction: Optional instruction replacing the default user text.

        Returns:
            PreparedItem or None if item has no images.
        """
        from PIL import Image as PILImage

        if not item.images:
            logger.warning("GlmOcrPreprocessor: item %d has no images", index)
            return None

        img_input = item.images[0]
        pil_img = PILImage.open(io.BytesIO(img_input["data"]))
        original_size = (pil_img.width, pil_img.height)

        if pil_img.mode != "RGB":
            pil_img = pil_img.convert("RGB")

        text = instruction or self._user_text
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": pil_img},
                    {"type": "text", "text": text},
                ],
            }
        ]

        inputs = self._processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )

        payload = GlmOcrPayload(
            inputs=dict(inputs),
            original_size=original_size,
        )

        return PreparedItem(payload=payload, cost=1, original_index=index)

    def prepare(
        self,
        items: list[Item],
        *,
        config: ModelConfig,
        is_query: bool = False,
        instruction: str | None = None,
        task: str | None = None,
    ) -> PreparedBatch[GlmOcrPayload]:
        """Preprocess images for GLM-OCR.

        Uses parallel processing for batches of 2+ images to utilize
        multiple CPU cores (PIL releases GIL during image operations).

        Args:
            items: Items with images field.
            config: Model config (unused - uses processor defaults).
            is_query: Whether items are queries (unused for GLM-OCR).
            instruction: Optional instruction replacing the default user text.
            task: Task token (unused for GLM-OCR).

        Returns:
            PreparedBatch with GlmOcrPayload items.
        """
        if len(items) == 1:
            result = self._process_single_image(items[0], 0, instruction=instruction)
            if result is None:
                return PreparedBatch(items=[], total_cost=0, modality="image")
            return PreparedBatch(items=[result], total_cost=1, modality="image")

        executor = get_image_executor()
        futures = [executor.submit(self._process_single_image, item, i, instruction) for i, item in enumerate(items)]

        prepared_items: list[PreparedItem[GlmOcrPayload]] = []
        for future in futures:
            result = future.result()
            if result is not None:
                prepared_items.append(result)

        return PreparedBatch(
            items=prepared_items,
            total_cost=len(prepared_items),
            modality="image",
        )

    def collate(
        self,
        prepared: list[PreparedItem[GlmOcrPayload]],
        *,
        device: str,
        dtype: Any = None,
    ) -> dict[str, Any]:
        """Collate GLM-OCR items into model-ready tensors.

        GLM-OCR processes one image at a time: its pixel_values tensor is
        a flattened patch sequence with no batch dim, and other keys
        already carry a batch dim of 1 from apply_chat_template.

        Args:
            prepared: List of prepared items (single-item batches).
            device: Target device.
            dtype: Optional dtype for floating-point tensors.

        Returns:
            Dict with input tensors for the model.
        """
        if not prepared:
            return {}

        out: dict[str, Any] = {}
        for k, v in prepared[0].payload.inputs.items():
            if hasattr(v, "is_floating_point") and v.is_floating_point() and dtype is not None:
                out[k] = v.to(device=device, dtype=dtype)
            elif hasattr(v, "to"):
                out[k] = v.to(device)
            else:
                out[k] = v
        return out


class DetectionPreprocessor:
    """Preprocessor for object detection models (GroundingDINO, OWL-v2).

    Handles CPU-bound image preprocessing:
    1. Load image from bytes (PIL)
    2. Convert to RGB if needed
    3. Run image_processor to produce tensors (resize, normalize)
    4. Store original size for bbox denormalization

    This moves heavy image processing (resize, normalize) off the GPU thread,
    allowing overlap with inference on previous batches. The adapter only needs
    to run the tokenizer with text labels at inference time.

    Key insight: HuggingFace processors for detection models have separate
    `image_processor` and `tokenizer` components:
    - `processor.image_processor(images=...)` -> pixel_values tensor (NO labels needed)
    - `processor.tokenizer(text=...)` -> input_ids (needs labels)

    This allows us to do 94% of preprocessing (image_processor: ~23ms) in the
    thread pool, leaving only tokenization (~0.1ms) for the inference thread.

    Thread-safe: PIL and HuggingFace image processors handle concurrent calls.

    Implements the Preprocessor protocol for integration with PreprocessorRegistry.
    """

    def __init__(
        self,
        image_processor: Any,
        model_name: str,
    ) -> None:
        """Initialize with a HuggingFace image processor.

        Args:
            image_processor: HuggingFace image processor (processor.image_processor).
            model_name: Model name for logging.
        """
        self._image_processor = image_processor
        self._model_name = model_name

    @property
    def modality(self) -> str:
        """Return 'image'."""
        return "image"

    def _process_single_image(
        self,
        item: Item,
        index: int,
    ) -> PreparedItem[DetectionPayload] | None:
        """Process a single image item (thread-safe).

        Args:
            item: Item with images field (ImageInput format).
            index: Original index for reordering.

        Returns:
            PreparedItem or None if item has no images.
        """
        from PIL import Image as PILImage

        from sie_server.types.inputs import is_image_input

        if not item.images:
            logger.warning("DetectionPreprocessor: item %d has no images", index)
            return None

        img = item.images[0]
        if not is_image_input(img):
            logger.warning("DetectionPreprocessor: item %d has non-ImageInput image", index)
            return None

        # Load image from bytes - PIL releases GIL during decode
        pil_img = PILImage.open(io.BytesIO(img["data"]))
        original_size = (pil_img.width, pil_img.height)

        # Convert to RGB if needed
        if pil_img.mode != "RGB":
            pil_img = pil_img.convert("RGB")

        # Run image_processor to produce tensor (resize, normalize)
        # This is the expensive part (~23ms, 94% of preprocessing)
        processed = self._image_processor(images=pil_img, return_tensors="pt")
        pixel_values = processed["pixel_values"].squeeze(0)  # Remove batch dim [C, H, W]

        payload = DetectionPayload(
            pixel_values=pixel_values,  # torch.Tensor [C, H, W]
            original_size=original_size,
        )

        return PreparedItem(payload=payload, cost=1, original_index=index)

    def prepare(
        self,
        items: list[Item],
        *,
        config: ModelConfig,
        is_query: bool = False,
        instruction: str | None = None,
        task: str | None = None,
    ) -> PreparedBatch[DetectionPayload]:
        """Preprocess images for detection.

        Uses parallel processing for batches of 2+ images to utilize
        multiple CPU cores (PIL releases GIL during image operations).

        Args:
            items: Items with images field (ImageInput format).
            config: Model config (unused).
            is_query: Whether items are queries (unused for detection).
            instruction: Optional instruction (unused for detection).
            task: Optional task token (unused for detection).

        Returns:
            PreparedBatch with DetectionPayload items containing PIL images.
        """
        # Single item: process directly (no thread pool overhead)
        if len(items) == 1:
            result = self._process_single_image(items[0], 0)
            if result is None:
                return PreparedBatch(items=[], total_cost=0, modality="image")
            return PreparedBatch(items=[result], total_cost=1, modality="image")

        # Multiple items: parallel processing
        executor = get_image_executor()
        futures = [executor.submit(self._process_single_image, item, i) for i, item in enumerate(items)]

        # Collect results in order
        prepared_items: list[PreparedItem[DetectionPayload]] = []
        for future in futures:
            result = future.result()
            if result is not None:
                prepared_items.append(result)

        return PreparedBatch(
            items=prepared_items,
            total_cost=len(prepared_items),
            modality="image",
        )

    def collate(
        self,
        prepared: list[PreparedItem[DetectionPayload]],
        *,
        device: str,
    ) -> dict[str, Any]:
        """Collate detection items into batched tensor.

        Stacks preprocessed image tensors and collects original sizes
        for bbox denormalization during post-processing.

        Args:
            prepared: List of prepared items with pixel_values tensors.
            device: Target device for tensors.

        Returns:
            Dict with 'pixel_values' tensor [B, C, H, W] and 'original_sizes' list.
        """
        import torch

        if not prepared:
            return {
                "pixel_values": torch.tensor([]),
                "original_sizes": [],
            }

        # Stack pixel values into batch tensor
        pixel_values = torch.stack([p.payload.pixel_values for p in prepared])
        original_sizes = [p.payload.original_size for p in prepared]

        return {
            "pixel_values": pixel_values.to(device),
            "original_sizes": original_sizes,
        }


# Canonical task -> prompt mapping from the PaddleOCR-VL-1.5 model card
# (PROMPTS dict in the README; trailing colons included). Keep in sync with
# PaddleOCRVLAdapter._VALID_TASKS.
_PADDLEOCR_VL_TASK_PROMPTS: dict[str, str] = {
    "ocr": "OCR:",
    "table": "Table Recognition:",
    "formula": "Formula Recognition:",
    "chart": "Chart Recognition:",
    "spotting": "Spotting:",
    "seal": "Seal Recognition:",
}


class PaddleOCRVLPreprocessor:
    """Preprocessor for PaddleOCR-VL document OCR model.

    Builds a chat-template prompt keyed by task (ocr/table/formula/chart/
    spotting/seal) and runs the HuggingFace processor to produce pixel_values,
    input_ids, attention_mask, and image_grid_thw.

    Thread-safe: PIL and HuggingFace processors handle concurrent calls.
    """

    def __init__(
        self,
        processor: Any,
        model_name: str,
        *,
        default_task: str = "ocr",
    ) -> None:
        if default_task not in _PADDLEOCR_VL_TASK_PROMPTS:
            msg = f"default_task {default_task!r} must be one of {tuple(_PADDLEOCR_VL_TASK_PROMPTS)}"
            raise ValueError(msg)
        self._processor = processor
        self._model_name = model_name
        self._default_task = default_task

    @property
    def modality(self) -> str:
        return "image"

    def _task_prompt(self, task: str | None, instruction: str | None) -> str:
        """Resolve the text prompt for the user message.

        Instruction overrides the task-based prompt when provided.
        """
        if instruction:
            return instruction
        resolved_task = task or self._default_task
        return _PADDLEOCR_VL_TASK_PROMPTS.get(resolved_task, _PADDLEOCR_VL_TASK_PROMPTS[self._default_task])

    def _build_messages(self, prompt_text: str) -> list[dict[str, Any]]:
        return [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": prompt_text},
                ],
            }
        ]

    def _process_single_image(
        self,
        item: Item,
        index: int,
        text: str,
    ) -> PreparedItem[PaddleOCRVLPayload] | None:
        from PIL import Image as PILImage

        if not item.images:
            logger.warning("PaddleOCRVLPreprocessor: item %d has no images", index)
            return None

        img_input = item.images[0]
        pil_img = PILImage.open(io.BytesIO(img_input["data"]))
        original_size = (pil_img.width, pil_img.height)

        if pil_img.mode != "RGB":
            pil_img = pil_img.convert("RGB")

        inputs = self._processor(
            text=text,
            images=[pil_img],
            return_tensors="pt",
        )

        payload = PaddleOCRVLPayload(
            pixel_values=inputs["pixel_values"],
            input_ids=inputs["input_ids"].squeeze(0),
            attention_mask=inputs["attention_mask"].squeeze(0),
            image_grid_thw=inputs["image_grid_thw"],
            original_size=original_size,
        )
        return PreparedItem(payload=payload, cost=1, original_index=index)

    def prepare(
        self,
        items: list[Item],
        *,
        config: ModelConfig,
        is_query: bool = False,
        instruction: str | None = None,
        task: str | None = None,
    ) -> PreparedBatch[PaddleOCRVLPayload]:
        prompt_text = self._task_prompt(task, instruction)
        messages = self._build_messages(prompt_text)
        text = self._processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False,
        )

        if len(items) == 1:
            result = self._process_single_image(items[0], 0, text)
            if result is None:
                return PreparedBatch(items=[], total_cost=0, modality="image")
            return PreparedBatch(items=[result], total_cost=1, modality="image")

        executor = get_image_executor()
        futures = [executor.submit(self._process_single_image, item, i, text) for i, item in enumerate(items)]

        prepared_items: list[PreparedItem[PaddleOCRVLPayload]] = []
        for future in futures:
            result = future.result()
            if result is not None:
                prepared_items.append(result)

        return PreparedBatch(
            items=prepared_items,
            total_cost=len(prepared_items),
            modality="image",
        )

    def collate(
        self,
        prepared: list[PreparedItem[PaddleOCRVLPayload]],
        *,
        device: str,
        dtype: Any = None,
    ) -> dict[str, Any]:
        """Collate items into batched tensors.

        PaddleOCR-VL processes one image at a time (variable-size inputs via NaViT
        patch packing); batching multiple images requires careful grid handling
        that the model's own path does not expose. This collate emits the first
        item's tensors, matching LightOnOCR's approach.
        """
        import torch

        if not prepared:
            return {
                "pixel_values": torch.tensor([]),
                "input_ids": torch.tensor([]),
                "attention_mask": torch.tensor([]),
                "image_grid_thw": torch.tensor([]),
            }

        p = prepared[0]
        pixel_values = p.payload.pixel_values
        if dtype is not None:
            pixel_values = pixel_values.to(dtype=dtype)

        return {
            "pixel_values": pixel_values.to(device),
            "input_ids": p.payload.input_ids.unsqueeze(0).to(device),
            "attention_mask": p.payload.attention_mask.unsqueeze(0).to(device),
            "image_grid_thw": p.payload.image_grid_thw.to(device),
        }
