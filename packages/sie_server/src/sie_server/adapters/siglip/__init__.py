"""SigLIP model adapter for image-text embedding.

This adapter provides support for SigLIP (Sigmoid Loss for Language Image Pre-training)
models that produce aligned embeddings for both images and text in a shared vector space.

Per roadmap Project 10.4, uses transformers SiglipModel with SiglipProcessor
for Phase 1. SigLIP differs from CLIP in using sigmoid loss instead of softmax
and not having a separate projection_dim - it uses hidden_size directly.

Supports two backends:

- ``transformers`` (default): standard ``SiglipModel`` + ``SiglipProcessor``
  for stock checkpoints such as ``google/siglip-*``.
- ``open_clip``: native ``open_clip`` loading for SigLIP-architecture
  checkpoints distributed as open_clip weights, e.g.
  ``Marqo/marqo-ecommerce-embeddings-B``. This bypasses the model's HF
  custom-code wrapper, which fails to load under newer transformers because
  the wrapper instantiates real-weight submodules inside ``__init__`` while
  ``from_pretrained`` is using a meta-tensor init context.

Supports:
- Text-only encoding → dense embeddings
- Image-only encoding → dense embeddings
- Image+text encoding → image embeddings (for retrieval)

Example configuration (transformers backend):
    SiglipAdapter(
        model_name_or_path="google/siglip-so400m-patch14-384",
    )

Example configuration (open_clip backend):
    SiglipAdapter(
        model_name_or_path="Marqo/marqo-ecommerce-embeddings-B",
        backend="open_clip",
        open_clip_model_id="hf-hub:Marqo/marqo-ecommerce-embeddings-B",
        dense_dim=768,
    )
"""

from __future__ import annotations

import io
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar, Literal

import torch

from sie_server.adapters._base_adapter import BaseAdapter
from sie_server.adapters._spec import AdapterSpec
from sie_server.adapters._types import ComputePrecision
from sie_server.core.inference_output import EncodeOutput

if TYPE_CHECKING:
    from PIL import Image

    from sie_server.types.inputs import Item

logger = logging.getLogger(__name__)

# Error messages
_ERR_NO_INPUT = "SiglipAdapter requires either text or images input"
_ERR_OPEN_CLIP_ID = (
    "SiglipAdapter(backend='open_clip') requires open_clip_model_id (e.g. 'hf-hub:Marqo/marqo-ecommerce-embeddings-B')"
)
_ERR_OPEN_CLIP_DIM = (
    "SiglipAdapter(backend='open_clip') requires an explicit dense_dim "
    "(open_clip does not surface embedding dim through a single config attribute)"
)


SiglipBackend = Literal["transformers", "open_clip"]


class SiglipAdapter(BaseAdapter):
    """Adapter for SigLIP image-text embedding models.

    Supports encoding text, images, or both into dense embeddings in a shared
    vector space. Uses HuggingFace transformers SiglipModel and SiglipProcessor
    by default; can be switched to the ``open_clip`` library for checkpoints
    distributed in that format.

    Key difference from CLIP: SigLIP uses hidden_size directly instead of
    projection_dim for the embedding dimension.
    """

    spec: ClassVar[AdapterSpec] = AdapterSpec(
        inputs=("text", "image"),
        outputs=("dense",),
        unload_fields=(
            "_model",
            "_processor",
            "_dense_dim",
            "_open_clip_preprocess",
            "_open_clip_tokenizer",
        ),
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
        backend: SiglipBackend = "transformers",
        open_clip_model_id: str | None = None,
        dense_dim: int | None = None,
    ) -> None:
        """Initialize the adapter.

        Args:
            model_name_or_path: HuggingFace model ID or local path. Used by
                the ``transformers`` backend; informational only for the
                ``open_clip`` backend (used in logs and as the
                ``ImagePreprocessor`` model name).
            normalize: Whether to L2-normalize embeddings.
            compute_precision: Compute precision for inference.
            trust_remote_code: Forwarded to ``SiglipModel.from_pretrained`` /
                ``SiglipProcessor.from_pretrained`` in the transformers
                backend. Ignored by the ``open_clip`` backend.
            max_seq_length: Ignored - SigLIP uses fixed token length from model config.
            backend: Which loader to use. ``transformers`` (default) for stock
                SigLIP checkpoints; ``open_clip`` for open_clip-distributed
                SigLIP checkpoints (e.g. Marqo).
            open_clip_model_id: open_clip model identifier (e.g.
                ``"hf-hub:Marqo/marqo-ecommerce-embeddings-B"``). Required
                when ``backend="open_clip"``.
            dense_dim: Optional explicit embedding dimension. Required for
                the ``open_clip`` backend (open_clip does not surface dim via
                a single config attribute). When omitted with the
                ``transformers`` backend, the adapter reads
                ``vision_config.hidden_size`` from the loaded model config.
        """
        self._model_name_or_path = str(model_name_or_path)
        self._normalize = normalize
        self._compute_precision = compute_precision
        self._trust_remote_code = trust_remote_code
        self._backend: SiglipBackend = backend
        self._open_clip_model_id = open_clip_model_id
        self._dense_dim_override = dense_dim

        # Validate the open_clip backend's required options up-front so the
        # error surfaces at adapter construction rather than at load time.
        if self._backend == "open_clip":
            if not self._open_clip_model_id:
                raise ValueError(_ERR_OPEN_CLIP_ID)
            if self._dense_dim_override is None:
                raise ValueError(_ERR_OPEN_CLIP_DIM)

        self._model: Any | None = None
        self._processor: Any | None = None
        self._open_clip_preprocess: Any | None = None
        self._open_clip_tokenizer: Any | None = None
        self._device: str | None = None
        self._dense_dim: int | None = None

    def load(self, device: str) -> None:
        """Load the model onto the specified device.

        Args:
            device: Device string (e.g., "cuda:0", "cpu").
        """
        self._device = device

        # Determine dtype
        dtype = self._resolve_dtype()

        logger.info(
            "Loading SigLIP model %s on device=%s with dtype=%s backend=%s",
            self._model_name_or_path,
            device,
            dtype,
            self._backend,
        )

        if self._backend == "open_clip":
            self._load_open_clip(device, dtype)
        else:
            self._load_transformers(device, dtype)

        # Resolve embedding dimension. The ``open_clip`` backend always
        # supplies an explicit override (validated in ``__init__``); the
        # ``transformers`` backend can fall back to ``vision_config``.
        if self._dense_dim_override is not None:
            self._dense_dim = self._dense_dim_override
        else:
            assert self._backend == "transformers"
            assert self._model is not None  # guarded by _load_transformers
            vision_config = getattr(self._model.config, "vision_config", None)
            if vision_config is None or not hasattr(vision_config, "hidden_size"):
                msg = "Cannot infer dense_dim from model config; pass adapter_options.loadtime.dense_dim explicitly."
                raise RuntimeError(msg)
            self._dense_dim = vision_config.hidden_size

    def _load_transformers(self, device: str, dtype: torch.dtype) -> None:
        """Load via transformers SiglipModel/SiglipProcessor."""
        from transformers import SiglipModel, SiglipProcessor

        self._processor = SiglipProcessor.from_pretrained(
            self._model_name_or_path,
            trust_remote_code=self._trust_remote_code,
        )

        self._model = SiglipModel.from_pretrained(
            self._model_name_or_path,
            torch_dtype=dtype,
            trust_remote_code=self._trust_remote_code,
        )
        self._model.to(device)
        self._model.eval()

    def _load_open_clip(self, device: str, dtype: torch.dtype) -> None:
        """Load via open_clip native loader.

        The model is moved to ``device`` and cast to ``dtype`` after
        construction; this matches the transformers path's behavior and avoids
        the meta-tensor incompatibility that bites HF custom-code wrappers
        which call ``self.model.to(...)`` from inside ``__init__``.
        """
        import open_clip

        assert self._open_clip_model_id is not None  # guarded in __init__

        model, _train_preproc, val_preproc = open_clip.create_model_and_transforms(self._open_clip_model_id)
        tokenizer = open_clip.get_tokenizer(self._open_clip_model_id)

        model.to(device=device, dtype=dtype)
        model.eval()

        self._model = model
        self._open_clip_preprocess = val_preproc
        self._open_clip_tokenizer = tokenizer
        # ``_processor`` stays ``None`` on this backend; ``_check_loaded()``
        # only consults ``_model`` and the encode paths branch on
        # ``self._backend`` to pick the right tokenizer/preprocess callable.

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
            prepared_items: Optional list of ``PreparedItem[ImagePayload]``
                from the framework's image preprocessor. When supplied, the
                ``open_clip`` backend reuses the preprocessed pixel tensors
                (which were computed in parallel on a CPU executor thread)
                instead of re-running ``val_preproc`` here. The transformers
                backend currently ignores this hint.

        Returns:
            EncodeOutput with dense embeddings.
        """
        # ``_check_loaded`` guards on ``_model``; backend-specific helpers
        # (``_open_clip_tokenizer`` / ``_open_clip_preprocess`` for open_clip,
        # ``_processor`` for transformers) are validated inside the encode
        # branches via local ``assert``s.
        self._check_loaded()

        self._validate_output_types(output_types)

        # Index prepared image tensors by the original item index so the
        # per-item loop can reuse them. Only the open_clip backend currently
        # consumes these; the transformers ImagePreprocessor produces tensors
        # in a layout matching SiglipProcessor, but the existing transformers
        # code path re-runs the processor inline — left untouched here to
        # avoid behavioral drift on the google/siglip* regression tests.
        prepared_by_index: dict[int, Any] = {}
        if self._backend == "open_clip" and prepared_items:
            for prepared in prepared_items:
                payload = getattr(prepared, "payload", None)
                if payload is None:
                    continue
                pixel_values = getattr(payload, "pixel_values", None)
                if pixel_values is None:
                    continue
                prepared_by_index[prepared.original_index] = pixel_values

        # Encode each item individually and stack into batch
        import numpy as np

        embeddings_list = []
        for i, item in enumerate(items):
            embedding = self._encode_single_item(item, prepared_pixel_values=prepared_by_index.get(i))
            embeddings_list.append(embedding)

        # Stack into batched array [batch, dim]
        dense_batch = np.stack(embeddings_list, axis=0)

        return EncodeOutput(
            dense=dense_batch,
            batch_size=len(items),
            is_query=is_query,
            dense_dim=self._dense_dim,
        )

    def _encode_single_item(self, item: Item, prepared_pixel_values: Any = None) -> Any:
        """Encode a single item (text, image, or both).

        Args:
            item: Input item.
            prepared_pixel_values: Optional pre-processed ``[C, H, W]`` tensor
                from the framework's image preprocessor. Only honored on the
                ``open_clip`` backend and only when the item has exactly one
                image (matching what the preprocessor consumed). Falls back
                to inline preprocessing otherwise.

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
            # Image encoding (or image+text where image takes precedence).
            # The framework's preprocessor only handles the first image per
            # item, so we can only reuse it when the item has exactly one.
            if (
                self._backend == "open_clip"
                and prepared_pixel_values is not None
                and images is not None
                and len(images) == 1
            ):
                return self._encode_images([], prepared_pixel_values=prepared_pixel_values)
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

    def _encode_images(
        self,
        images: list[Image.Image],
        prepared_pixel_values: Any = None,
    ) -> Any:
        """Encode images into embeddings.

        Args:
            images: List of PIL Images. Ignored on the ``open_clip`` backend
                when ``prepared_pixel_values`` is supplied.
            prepared_pixel_values: Optional pre-processed ``[C, H, W]`` tensor
                from the framework's preprocessor. Only the ``open_clip``
                backend uses this fast path; when supplied, ``val_preproc``
                is skipped and the tensor is moved to device and batched
                directly.

        Returns:
            Numpy array of shape [dense_dim] (averaged if multiple images).
        """
        assert self._model is not None

        from torch.nn import functional

        if self._backend == "open_clip":
            # ``val_preproc`` and the framework preprocessor both return
            # float32 tensors. The open_clip model was cast to ``dtype`` at
            # load time (e.g. fp16 on CUDA), so we must match here — unlike
            # ``transformers.SiglipModel``, ``open_clip``'s ``encode_image``
            # does not auto-cast inputs.
            model_dtype = next(self._model.parameters()).dtype
            if prepared_pixel_values is not None:
                # Fast path: framework preprocessor already produced a
                # [C, H, W] tensor in the val_preproc layout. Add the batch
                # dim, move to device, and cast to the model's dtype.
                pixel_values = prepared_pixel_values.unsqueeze(0).to(device=self._device, dtype=model_dtype)
                effective_count = 1
            else:
                assert self._open_clip_preprocess is not None
                pixel_values = torch.stack([self._open_clip_preprocess(img) for img in images]).to(
                    device=self._device, dtype=model_dtype
                )
                effective_count = len(images)
            with torch.inference_mode():
                image_features = self._model.encode_image(pixel_values, normalize=self._normalize)
        else:
            assert self._processor is not None
            inputs = self._processor(images=images, return_tensors="pt")
            inputs = {k: v.to(self._device) for k, v in inputs.items()}
            with torch.inference_mode():
                image_features = self._model.get_image_features(**inputs)
                if self._normalize:
                    image_features = functional.normalize(image_features, p=2, dim=-1)
            effective_count = len(images)

        # If multiple images, average the embeddings
        if effective_count > 1:
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

        from torch.nn import functional

        if self._backend == "open_clip":
            assert self._open_clip_tokenizer is not None
            input_ids = self._open_clip_tokenizer([text]).to(self._device)
            with torch.inference_mode():
                text_features = self._model.encode_text(input_ids, normalize=self._normalize)
        else:
            assert self._processor is not None
            # Process text - use max_length padding to match MTEB behavior
            # SigLIP text embeddings depend on sequence length, so consistent padding is required
            inputs = self._processor(text=[text], return_tensors="pt", padding="max_length", truncation=True)
            inputs = {k: v.to(self._device) for k, v in inputs.items()}
            with torch.inference_mode():
                text_features = self._model.get_text_features(**inputs)
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
            ``ImagePreprocessor`` wrapping the SiglipProcessor on the
            transformers backend, ``OpenCLIPImagePreprocessor`` wrapping the
            open_clip ``val_preproc`` callable on the open_clip backend, or
            ``None`` if not loaded.
        """
        if self._backend == "open_clip":
            if self._open_clip_preprocess is None:
                return None
            from sie_server.core.preprocessor.image import OpenCLIPImagePreprocessor

            return OpenCLIPImagePreprocessor(self._open_clip_preprocess, self._model_name_or_path)

        if self._processor is None:
            return None

        from sie_server.core.preprocessor import ImagePreprocessor

        return ImagePreprocessor(self._processor, self._model_name_or_path)
