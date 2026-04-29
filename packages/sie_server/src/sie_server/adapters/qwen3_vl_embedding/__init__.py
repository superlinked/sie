from __future__ import annotations

import io
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar

import numpy as np
import torch
from torch.nn import functional as F

from sie_server.adapters._base_adapter import BaseAdapter
from sie_server.adapters._spec import AdapterSpec
from sie_server.adapters._types import ERR_NOT_LOADED, ComputePrecision
from sie_server.core.inference_output import EncodeOutput

if TYPE_CHECKING:
    from PIL import Image
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

    from sie_server.types.inputs import Item

logger = logging.getLogger(__name__)

_ERR_NO_INPUT = "Qwen3VLEmbeddingAdapter requires either text, images, or video input"

# Default system prompt used by Qwen3-VL-Embedding for instruction-aware retrieval.
_DEFAULT_INSTRUCTION = "Represent the user's input."

_SUPPORTED_POOLING = ("last", "mean")


def _build_conversation(
    *,
    text: str | None = None,
    images: list[Image.Image] | None = None,
    video_frames: list[Image.Image] | None = None,
    instruction: str = _DEFAULT_INSTRUCTION,
) -> list[dict[str, Any]]:
    """Build a chat conversation in the Qwen3-VL format.

    The model expects:
      system: <instruction>
      user: [optional images/video frames] <text>
    """
    content: list[dict[str, Any]] = []
    if images:
        for img in images:
            content.append({"type": "image", "image": img})
    if video_frames:
        for frame in video_frames:
            content.append({"type": "image", "image": frame})
    if text:
        content.append({"type": "text", "text": text})
    if not content:
        content.append({"type": "text", "text": ""})

    return [
        {"role": "system", "content": [{"type": "text", "text": instruction}]},
        {"role": "user", "content": content},
    ]


class Qwen3VLEmbeddingAdapter(BaseAdapter):
    """Adapter for Qwen3-VL-Embedding multimodal embedding models.

    Qwen3-VL-Embedding-2B uses the Qwen3-VL architecture to produce dense
    embeddings from text, images, or mixed inputs in a shared vector space.

    Key features:
    - 2048-dim dense embeddings (MRL: supports 64-2048 via truncation)
    - Chat-template-based instruction-aware encoding
    - Last-token and mean pooling from the causal decoder
    - Text + image + video inputs (video encoded as extracted frames)
    - Apache 2.0 license, 2B parameters, fits L4 with headroom

    Target models:
    - Qwen/Qwen3-VL-Embedding-2B
    - Qwen/Qwen3-VL-Embedding-8B (future)
    """

    spec: ClassVar[AdapterSpec] = AdapterSpec(
        inputs=("text", "image", "video"),
        outputs=("dense",),
        dense_dim=2048,
        unload_fields=("_model", "_processor", "_dense_dim"),
        default_preprocessor="image",
    )

    def __init__(
        self,
        model_name_or_path: str | Path,
        *,
        normalize: bool = True,
        compute_precision: ComputePrecision = "bfloat16",
        trust_remote_code: bool = False,
        max_seq_length: int | None = None,
        pooling: str = "last",
        default_instruction: str = _DEFAULT_INSTRUCTION,
    ) -> None:
        if pooling not in _SUPPORTED_POOLING:
            msg = f"Unsupported pooling '{pooling}', must be one of {_SUPPORTED_POOLING}"
            raise ValueError(msg)

        self._model_name_or_path = str(model_name_or_path)
        self._normalize = normalize
        self._compute_precision = compute_precision
        self._trust_remote_code = trust_remote_code
        self._max_seq_length = max_seq_length
        self._pooling = pooling
        self._default_instruction = default_instruction

        self._model: Qwen3VLForConditionalGeneration | None = None
        self._processor: AutoProcessor | None = None
        self._device: str | None = None
        self._dense_dim: int | None = 2048

    def load(self, device: str) -> None:
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

        self._device = device
        dtype = self._resolve_dtype()
        attn_impl = self._resolve_attn_implementation(device)

        logger.info(
            "Loading Qwen3-VL-Embedding %s on device=%s dtype=%s attn=%s pooling=%s max_seq_length=%s",
            self._model_name_or_path,
            device,
            dtype,
            attn_impl,
            self._pooling,
            self._max_seq_length,
        )

        proc_kwargs: dict[str, Any] = {
            "trust_remote_code": self._trust_remote_code,
            "min_pixels": 256 * 28 * 28,
            "max_pixels": 1280 * 28 * 28,
        }
        self._processor = AutoProcessor.from_pretrained(
            self._model_name_or_path,
            **proc_kwargs,
        )
        if self._max_seq_length is not None and hasattr(self._processor, "tokenizer"):
            self._processor.tokenizer.model_max_length = self._max_seq_length  # ty: ignore[unresolved-attribute]

        load_kwargs: dict[str, Any] = {
            "torch_dtype": dtype,
            "device_map": device,
            "trust_remote_code": self._trust_remote_code,
        }
        if attn_impl is not None:
            load_kwargs["attn_implementation"] = attn_impl

        self._model = Qwen3VLForConditionalGeneration.from_pretrained(
            self._model_name_or_path,
            **load_kwargs,
        )
        self._model.eval()

        # Qwen3VLConfig stores the text model hidden size under text_config
        cfg = self._model.config
        if hasattr(cfg, "hidden_size"):
            self._dense_dim = cfg.hidden_size
        elif hasattr(cfg, "text_config") and hasattr(cfg.text_config, "hidden_size"):
            self._dense_dim = cfg.text_config.hidden_size
        else:
            logger.warning("Could not determine hidden_size from config, defaulting to 2048")
            self._dense_dim = 2048

    def _resolve_dtype(self) -> torch.dtype:
        if not self._device or not str(self._device).startswith("cuda"):
            return torch.float32
        dtype_map = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }
        return dtype_map.get(self._compute_precision, torch.bfloat16)

    def _resolve_attn_implementation(self, device: str) -> str | None:
        if not device.startswith("cuda"):
            return None
        try:
            import flash_attn  # ty: ignore[unresolved-import]

            return "flash_attention_2"
        except (ImportError, RuntimeError) as exc:
            logger.info("flash_attn not available (%s), using sdpa attention", exc)
            return "sdpa"

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
        self._check_loaded()
        assert self._model is not None
        assert self._processor is not None

        inst = instruction or self._default_instruction

        embeddings_list: list[np.ndarray] = []
        for item in items:
            emb = self._encode_single_item(item, instruction=inst)
            embeddings_list.append(emb)

        # Free GPU memory once after the full batch rather than per item
        if self._device and self._device.startswith("cuda"):
            torch.cuda.empty_cache()

        dense_batch = np.stack(embeddings_list, axis=0)

        return EncodeOutput(
            dense=dense_batch,
            batch_size=len(items),
            is_query=is_query,
            dense_dim=self._dense_dim,
        )

    def _encode_single_item(self, item: Any, *, instruction: str) -> np.ndarray:
        has_text = item.text is not None
        has_images = item.images is not None and len(item.images) > 0
        has_video = item.video is not None

        if not has_text and not has_images and not has_video:
            raise ValueError(_ERR_NO_INPUT)

        # Load images (all of them, not just the first)
        pil_images: list[Image.Image] | None = None
        if has_images:
            pil_images = self._load_images(item)

        # Extract video frames as images
        video_frames: list[Image.Image] | None = None
        if has_video:
            video_frames = self._extract_video_frames(item)

        # If the input was visual-only but all images/frames failed to load,
        # reject rather than silently embedding an empty prompt.
        has_loaded_visuals = bool(pil_images) or bool(video_frames)
        if not has_text and not has_loaded_visuals:
            msg = (
                f"{_ERR_NO_INPUT}: item had visual inputs "
                f"(images={len(item.images or [])}, video={'yes' if has_video else 'no'}) "
                "but all failed to load"
            )
            raise ValueError(msg)

        conversation = _build_conversation(
            text=item.text,
            images=pil_images,
            video_frames=video_frames,
            instruction=instruction,
        )

        return self._forward_conversation(conversation)

    def _forward_conversation(self, conversation: list[dict[str, Any]]) -> np.ndarray:
        """Apply chat template, tokenize, run forward pass, extract embedding."""
        assert self._model is not None
        assert self._processor is not None

        # Apply the chat template to get prompt text
        prompt = self._processor.apply_chat_template(  # ty: ignore[unresolved-attribute]
            conversation,
            tokenize=False,
            add_generation_prompt=True,
        )

        # Extract images from conversation for the processor
        images = []
        for msg in conversation:
            content = msg.get("content", [])
            if isinstance(content, list):
                for part in content:
                    if isinstance(part, dict) and part.get("type") == "image":
                        img = part.get("image")
                        if img is not None:
                            images.append(img)

        # Tokenize with processor
        proc_kwargs: dict[str, Any] = {
            "text": [prompt],
            "return_tensors": "pt",
            "padding": True,
        }
        if self._max_seq_length is not None:
            proc_kwargs["truncation"] = True
            proc_kwargs["max_length"] = self._max_seq_length
        if images:
            proc_kwargs["images"] = images

        inputs = self._processor(**proc_kwargs)  # ty: ignore[call-non-callable]
        inputs = {k: v.to(self._device) for k, v in inputs.items()}

        with torch.inference_mode():
            outputs = self._model(
                **inputs,
                output_hidden_states=True,
                return_dict=True,
            )

        last_hidden = outputs.hidden_states[-1]  # (1, seq_len, hidden_dim)
        mask = inputs.get("attention_mask")

        if self._pooling == "mean":
            embedding = self._mean_pool(last_hidden, mask)
        else:
            # Default: last-token pooling
            embedding = self._last_token_pool(last_hidden, mask)

        # Normalize
        if self._normalize:
            embedding = F.normalize(embedding.unsqueeze(0), p=2, dim=-1).squeeze(0)

        result = embedding.float().cpu().numpy()

        # Free intermediate tensors (batch-level empty_cache is in encode())
        del outputs, inputs, last_hidden

        return result

    @staticmethod
    def _last_token_pool(last_hidden: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
        if mask is not None:
            seq_lengths = mask.sum(dim=1) - 1  # (batch,)
            return last_hidden[0, int(seq_lengths[0].item())]
        return last_hidden[0, -1]

    @staticmethod
    def _mean_pool(last_hidden: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
        if mask is not None:
            mask_expanded = mask.unsqueeze(-1).float()
            summed = (last_hidden * mask_expanded).sum(dim=1)
            counts = mask_expanded.sum(dim=1).clamp(min=1e-9)
            return (summed / counts).squeeze(0)
        return last_hidden.mean(dim=1).squeeze(0)

    # ------------------------------------------------------------------
    # Image / video loading
    # ------------------------------------------------------------------

    def _load_images(self, item: Any) -> list[Image.Image]:
        from PIL import Image

        pil_images = []
        for idx, img_input in enumerate(item.images or []):
            try:
                pil_img = Image.open(io.BytesIO(img_input["data"]))
                if pil_img.mode != "RGB":
                    pil_img = pil_img.convert("RGB")
                pil_images.append(pil_img)
            except (OSError, KeyError) as exc:
                logger.warning("Failed to load image %d: %s", idx, exc)
        return pil_images

    def _extract_video_frames(self, item: Any) -> list[Image.Image]:
        """Extract frames from video input and return as PIL images.

        **Limitation**: This is a placeholder that only handles image-decodable
        inputs (e.g. animated GIF first frame) via ``PIL.Image.open`` /
        ``io.BytesIO``.  True video files (mp4, webm, avi, etc.) are **not**
        decoded here and will return an empty list.  Proper video support
        requires an external decoding library (cv2, av, decord, etc.) to
        extract and sample frames before passing them to the model.
        """
        from PIL import Image

        video_input = item.video
        if video_input is None:
            return []

        try:
            video_bytes = video_input["data"]
        except (KeyError, TypeError) as exc:
            logger.warning("Failed to read video data: %s", exc)
            return []

        # Attempt to open as a PIL-decodable image (e.g. animated GIF).
        # Real video formats (mp4/webm) will fail here and return [].
        try:
            pil_img = Image.open(io.BytesIO(video_bytes))
            if pil_img.mode != "RGB":
                pil_img = pil_img.convert("RGB")
            return [pil_img]
        except (OSError, ValueError) as exc:
            logger.warning(
                "Could not decode video input as image via PIL.Image.open: %s "
                "(true video decoding requires cv2/av/decord)",
                exc,
            )
            return []

    def get_preprocessor(self) -> Any | None:
        # Qwen3-VL processor requires text alongside images (for chat template
        # token insertion). The generic ImagePreprocessor calls processor(images=...)
        # without text, which crashes Qwen3VLProcessor. Return None to use the
        # direct adapter call path where we handle the conversation template.
        return None
