"""CLIP model adapter for image-text embedding.

This adapter provides support for CLIP and similar image-text models that
produce aligned embeddings for both images and text in a shared vector space.

Uses transformers CLIPModel with CLIPProcessor. Optimization (FA2 varlen for
text) remains deferred.

Supports:
- Text-only encoding → dense embeddings
- Image-only encoding → dense embeddings
- Image+text encoding → concatenated/fused embeddings (model-dependent)

Example configuration:
    CLIPAdapter(
        model_name_or_path="openai/clip-vit-base-patch32",
    )
"""

from __future__ import annotations

import io
import logging
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar

import torch
from PIL import Image
from torch.nn import functional

from sie_server.adapters._base_adapter import BaseAdapter
from sie_server.adapters._spec import AdapterSpec
from sie_server.adapters._types import ERR_NOT_LOADED, ComputePrecision
from sie_server.core.inference_output import EncodeOutput
from sie_server.types.inputs import media_bytes

if TYPE_CHECKING:
    from transformers import CLIPModel, CLIPProcessor

    from sie_server.types.inputs import Item

logger = logging.getLogger(__name__)

# Error messages
_ERR_NO_INPUT = "CLIPAdapter requires either text or images input"

# Cross-item preprocessing is CPU-bound (PIL decode + resize). Fan the flat
# image set across a small bounded thread pool so decode/resize overlaps —
# PIL and the HF image processor release the GIL on the heavy numpy/C work.
_MAX_PREPROCESS_THREADS = 8


def _feature_tensor(out: Any) -> torch.Tensor:
    """Return the projected feature tensor from a ``get_*_features`` result.

    transformers <= 4.57.x (the Modal lane pin) returns the projected
    ``[B, dim]`` feature tensor directly; transformers 5.x wraps it in a
    ``BaseModelOutputWithPooling`` whose ``pooler_output`` is that *same*
    projected tensor (verified on 5.3.0: image/text ``pooler_output`` is
    ``[B, 512]`` = ``projection_dim``, while ``last_hidden_state`` is the
    un-projected hidden state). Unwrap so the adapter is robust across the pin
    instead of raising ``AttributeError`` in ``functional.normalize``.
    """
    if isinstance(out, torch.Tensor):
        return out
    pooled = getattr(out, "pooler_output", None)
    if isinstance(pooled, torch.Tensor):
        return pooled
    msg = f"unexpected get_*_features return type: {type(out).__name__}"
    raise TypeError(msg)


class CLIPAdapter(BaseAdapter):
    """Adapter for CLIP image-text embedding models.

    Supports encoding text, images, or both into dense embeddings in a shared
    vector space. Uses HuggingFace transformers CLIPModel and CLIPProcessor.
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
        revision: str | None = None,
        dense_dim: int | None = None,
    ) -> None:
        """Initialize the adapter.

        Args:
            model_name_or_path: HuggingFace model ID or local path.
            normalize: Whether to L2-normalize embeddings (CLIP typically uses normalized).
            compute_precision: Compute precision for inference.
            trust_remote_code: Whether to trust remote code (False for standard CLIP).
            max_seq_length: Ignored - CLIP uses fixed token length from model config.
            revision: Optional HuggingFace revision/branch/commit SHA to pin when
                loading the processor and model. If None, the default branch is
                used. Forwarded to ``from_pretrained(..., revision=...)``.
            dense_dim: Catalog-declared dense embedding dimension. If provided,
                validated against the loaded model's projection dimension.
        """
        self._model_name_or_path = str(model_name_or_path)
        self._normalize = normalize
        self._compute_precision = compute_precision
        self._trust_remote_code = trust_remote_code
        self._revision = revision
        self._configured_dense_dim = dense_dim

        self._model: CLIPModel | None = None
        self._processor: CLIPProcessor | None = None
        self._device: str | None = None
        self._dense_dim: int | None = dense_dim
        # HF fast tokenizers are NOT thread-safe: applying per-call
        # padding/truncation flags reconfigures the underlying Rust tokenizer,
        # which takes a mutable borrow. Concurrent text encodes (e.g. the
        # managed lane's ``@modal.concurrent`` inputs dispatched to a thread
        # pool) otherwise race with ``RuntimeError: Already borrowed``.
        # Serialise the tokenizer call — microseconds vs the GPU forward.
        self._tokenizer_lock = threading.Lock()
        # Lazily-created bounded pool used to overlap per-image CPU
        # preprocessing (decode + resize) within a batched encode call.
        # Guards lazy pool creation against concurrent first callers (same
        # hazard class as ``_tokenizer_lock``) so a race can't orphan a pool.
        self._preprocess_pool_lock = threading.Lock()
        self._preprocess_pool: ThreadPoolExecutor | None = None

    def load(self, device: str) -> None:
        """Load the model onto the specified device.

        Args:
            device: Device string (e.g., "cuda:0", "cpu").
        """
        from transformers import CLIPModel, CLIPProcessor

        self._device = device

        # Determine dtype
        dtype = self._resolve_dtype()

        logger.info(
            "Loading CLIP model %s on device=%s with dtype=%s",
            self._model_name_or_path,
            device,
            dtype,
        )

        shared_kwargs: dict[str, Any] = {"trust_remote_code": self._trust_remote_code}
        if self._revision is not None:
            shared_kwargs["revision"] = self._revision

        # Load processor (handles both text tokenization and image preprocessing)
        self._processor = CLIPProcessor.from_pretrained(
            self._model_name_or_path,
            **shared_kwargs,
        )

        # Load model
        self._model = CLIPModel.from_pretrained(
            self._model_name_or_path,
            torch_dtype=dtype,
            **shared_kwargs,
        )
        self._model.to(device)
        self._model.eval()

        # CLIP uses projection_dim for the aligned embedding space.
        self._dense_dim = self._validate_or_set_dense_dim(self._model.config.projection_dim)

    def _validate_or_set_dense_dim(self, observed_dim: int) -> int:
        """Validate observed CLIP projection width against configured dense_dim."""
        if self._configured_dense_dim is not None and observed_dim != self._configured_dense_dim:
            msg = (
                "CLIP embedding dimension mismatch: "
                f"configured dense_dim={self._configured_dense_dim}, model projection_dim={observed_dim}"
            )
            raise ValueError(msg)
        return observed_dim

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

        CLIP can encode text, images, or both. For items with only text,
        returns text embeddings. For items with only images, returns image
        embeddings. For items with both, returns image embeddings (common
        for image-text retrieval where images are the documents).

        Items are partitioned by modality and each modality is run as ONE
        stacked GPU forward (image tower and text tower). Semantics are
        preserved exactly per item: an item carrying more than one image is
        mean-pooled to a single vector (after per-image L2-normalize when
        ``normalize`` is set), and output order matches input order.

        Args:
            items: List of items to encode (with text and/or images).
            output_types: Which outputs to return (only "dense" supported).
            instruction: Optional instruction (not used by standard CLIP).
            is_query: Whether items are queries (affects nothing for base CLIP).
            prepared_items: Not used by this adapter.

        Returns:
            EncodeOutput with dense embeddings.
        """
        self._check_loaded()
        if self._processor is None:
            raise RuntimeError(ERR_NOT_LOADED)

        self._validate_output_types(output_types)

        import numpy as np

        # Partition items by modality (image takes precedence over text),
        # preserving original positions so the stacked output stays in order.
        image_indices: list[int] = []
        text_indices: list[int] = []
        for i, item in enumerate(items):
            has_text = item.text is not None
            images = item.images
            has_images = images is not None and len(images) > 0
            if not has_text and not has_images:
                raise ValueError(_ERR_NO_INPUT)
            if has_images:
                image_indices.append(i)
            else:
                text_indices.append(i)

        embeddings: list[Any] = [None] * len(items)

        if image_indices:
            image_vecs = self._encode_image_items([items[i] for i in image_indices])
            for slot, i in enumerate(image_indices):
                embeddings[i] = image_vecs[slot]

        per_item_token_counts: list[int] | None = None
        if text_indices:
            text_vecs, text_counts = self._encode_texts([items[i].text for i in text_indices])  # ty: ignore[invalid-argument-type]
            for slot, i in enumerate(text_indices):
                embeddings[i] = text_vecs[slot]
            # Scatter the per-text token counts back to their item positions
            # (image items → 0 text tokens), aligned 1:1 with ``items``.
            if isinstance(text_counts, list) and len(text_counts) == len(text_indices):
                per_item_token_counts = [0] * len(items)
                for slot, i in enumerate(text_indices):
                    per_item_token_counts[i] = text_counts[slot]

        # Stack into batched array [batch, dim]
        dense_batch = np.stack(embeddings, axis=0)

        output = EncodeOutput(
            dense=dense_batch,
            batch_size=len(items),
            is_query=is_query,
            dense_dim=self._dense_dim,
        )
        # Unit-meter seam (§7.3): stamp exact per-item TEXT-tower token counts.
        # Image items took the image tower (metered per image via the
        # ``count_input_images`` hook) and contribute 0 text tokens; only stamp
        # when at least one text item was encoded so a pure-image batch stays on
        # the image dimension. The encode pipeline forwards ``extra`` to the
        # result path for metering (§P3.5), in preference to re-tokenizing raw
        # text (which also misses the open_clip / mixed-batch cases).
        if per_item_token_counts is not None:
            output.extra["input_token_counts"] = per_item_token_counts
        # Modality routing as MEASURED (#2538) — see the SigLIP twin. An item
        # carrying both text and images takes the image tower only, so its exact
        # text-token count is zero; the pipeline reads this as the last-resort
        # basis when no tokenizer could produce counts for the batch.
        output.extra["text_tower_skipped"] = [i in set(image_indices) for i in range(len(items))]
        return output

    def _get_preprocess_pool(self) -> ThreadPoolExecutor:
        """Return the lazily-created bounded preprocessing thread pool."""
        pool = self._preprocess_pool
        if pool is None:
            with self._preprocess_pool_lock:
                pool = self._preprocess_pool
                if pool is None:
                    workers = min(_MAX_PREPROCESS_THREADS, os.cpu_count() or 1)
                    pool = ThreadPoolExecutor(max_workers=max(1, workers), thread_name_prefix="clip-preproc")
                    self._preprocess_pool = pool
        return pool

    def _preprocess_one(self, img_input: Any) -> torch.Tensor:
        """Decode + preprocess a single image input into a ``[C, H, W]`` tensor.

        Stateless per call — the HF image processor and PIL release the GIL on
        the heavy work, so this is safe to fan across the preprocessing pool.
        """
        assert self._processor is not None

        img_bytes = media_bytes(img_input, kind="image")
        pil_img = Image.open(io.BytesIO(img_bytes))
        # Convert to RGB if necessary (CLIP expects RGB)
        if pil_img.mode != "RGB":
            pil_img = pil_img.convert("RGB")
        processed = self._processor(images=pil_img, return_tensors="pt")
        return processed["pixel_values"][0]

    def _preprocess_image_batch(self, img_inputs: list[Any]) -> torch.Tensor:
        """Preprocess a flat list of image inputs into a ``[N, C, H, W]`` tensor.

        Per-image preprocessing is independent, so a single stacked tensor is
        bit-identical to preprocessing each image on its own — it is only fanned
        across threads to overlap the CPU-bound decode/resize.
        """
        if len(img_inputs) == 1:
            tensors = [self._preprocess_one(img_inputs[0])]
        else:
            pool = self._get_preprocess_pool()
            tensors = list(pool.map(self._preprocess_one, img_inputs))
        return torch.stack(tensors, dim=0)

    def _encode_image_items(self, items: list[Item]) -> Any:
        """Encode image items as one stacked forward, mean-pooling per item.

        Returns a ``[len(items), dim]`` float32 array, one row per item.
        """
        assert self._model is not None

        # Flatten every image across the items, remembering per-item counts so
        # the stacked features can be split back and mean-pooled per item.
        counts = [len(item.images or []) for item in items]
        flat_inputs = [img for item in items for img in (item.images or [])]

        pixel_values = self._preprocess_image_batch(flat_inputs).to(self._device)

        with torch.inference_mode():
            image_features = _feature_tensor(self._model.get_image_features(pixel_values=pixel_values))
            # L2 normalize per image, before any mean-pool (matches serial).
            if self._normalize:
                image_features = functional.normalize(image_features, p=2, dim=-1)

        # Split by per-item image counts; mean-pool multi-image items (the
        # mean runs on-device in the model dtype, exactly as the serial path).
        vecs = []
        offset = 0
        for count in counts:
            group = image_features[offset : offset + count]
            offset += count
            vecs.append(group[0] if count == 1 else group.mean(dim=0))

        return torch.stack(vecs, dim=0).float().cpu().numpy()

    def _encode_texts(self, texts: list[str]) -> tuple[Any, list[int] | None]:
        """Encode a list of texts as one stacked forward.

        Returns ``(embeddings, token_counts)``: a ``[len(texts), dim]`` float32
        array (one row per text) and the exact per-text token counts the text
        tower encoded (§7.3), or ``None`` when the tokenizer could not surface a
        clean count.
        """
        assert self._model is not None
        assert self._processor is not None

        # Process text as one batch. The HF fast tokenizer is not thread-safe
        # under the per-call padding/truncation reconfiguration, so serialise
        # the single batched call (microseconds vs the GPU forward).
        with self._tokenizer_lock:
            inputs = self._processor(text=list(texts), return_tensors="pt", padding=True, truncation=True)
            # Unit-meter seam (§7.3): the exact per-text token counts — post
            # truncation at the model context window (CLIP 77), special tokens
            # included — the tower actually encoded. Reuses the shared base
            # counter over the processor's own tokenizer (padding adds no
            # billable content, so a content-only recount equals the forward's
            # real length). ``None`` on any quirk leaves the meter on its reserve.
            token_counts = self._token_counts_or_none(
                self._processor.tokenizer,  # ty: ignore[unresolved-attribute]
                list(texts),
                expected_len=len(texts),
            )
        inputs = {k: v.to(self._device) for k, v in inputs.items()}

        with torch.inference_mode():
            text_features = _feature_tensor(self._model.get_text_features(**inputs))
            if self._normalize:
                text_features = functional.normalize(text_features, p=2, dim=-1)

        return text_features.float().cpu().numpy(), token_counts

    def _encode_text(self, text: str) -> Any:
        """Encode a single text into a ``[dim]`` embedding (thin batch wrapper)."""
        return self._encode_texts([text])[0][0]

    def _validate_output_types(self, output_types: list[str]) -> None:
        """Validate that output types are supported."""
        unsupported = set(output_types) - {"dense"}
        if unsupported:
            msg = f"Unsupported output types: {unsupported}. CLIP only supports 'dense'."
            raise ValueError(msg)

    def get_preprocessor(self) -> Any | None:
        """Return an ImagePreprocessor for CPU/GPU overlap.

        Returns:
            ImagePreprocessor wrapping the CLIPProcessor, or None if not loaded.
        """
        if self._processor is None:
            return None

        from sie_server.core.preprocessor import ImagePreprocessor

        return ImagePreprocessor(self._processor, self._model_name_or_path)

    def unload(self) -> None:
        """Shut down the preprocessing pool, then unload model weights."""
        pool = self._preprocess_pool
        self._preprocess_pool = None
        if pool is not None:
            pool.shutdown(wait=False)
        super().unload()
