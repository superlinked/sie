from __future__ import annotations

import io
import logging
import threading
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
from torch.nn import functional as F

from sie_server.adapters._base_adapter import BaseAdapter
from sie_server.adapters._multivector import maxsim_scores
from sie_server.adapters._spec import AdapterSpec
from sie_server.adapters._types import ComputePrecision
from sie_server.adapters._vision_patch_embed import rebind_vision_patch_embed
from sie_server.core.inference_output import EncodeOutput
from sie_server.core.postprocessor import MuveraConfig, MuveraPostprocessor
from sie_server.types.inputs import media_bytes

if TYPE_CHECKING:
    from PIL import Image as PILImage

    from sie_server.types.inputs import Item

logger = logging.getLogger(__name__)

_ERR_NO_INPUT = "ColQwen3Adapter requires either text or images input"


class ColQwen3Adapter(BaseAdapter):
    """Adapter for ColQwen3-style visual document retrieval models.

    ColQwen3 encodes document page images into multi-vector representations
    (320-dim per token) for late interaction retrieval. Built on Qwen3-VL,
    with a custom projection layer wrapper that exposes ``out.embeddings``.

    Target model: ``TomoroAI/tomoro-colqwen3-embed-4b`` (4B params).

    Loaded via ``AutoModel`` + ``AutoProcessor`` with ``trust_remote_code``
    because the model ships its own ``ColQwen3`` / ``ColQwen3Processor``
    classes (not in native transformers).
    """

    spec = AdapterSpec(
        inputs=("text", "image"),
        outputs=("multivector", "score"),
        multivector_dim=320,
        unload_fields=("_model", "_processor"),
        default_preprocessor="image",
    )

    def __init__(
        self,
        model_name_or_path: str | Path,
        *,
        normalize: bool = True,
        compute_precision: ComputePrecision = "bfloat16",
        trust_remote_code: bool = True,
        revision: str | None = None,
        max_seq_length: int | None = None,
        muvera_config: dict[str, Any] | None = None,
        token_dim: int = 320,
        max_num_visual_tokens: int = 1280,
    ) -> None:
        """Initialize the adapter.

        Args:
            model_name_or_path: HuggingFace model ID or local path.
            normalize: Whether to L2-normalize embeddings (the model's wrapper
                already normalizes; kept for interface parity).
            compute_precision: Compute precision for inference.
            trust_remote_code: Required for ColQwen3 (custom processor + model classes).
            revision: Optional HuggingFace revision/branch/commit SHA to pin when
                loading model artifacts. Forwarded to ``from_pretrained(..., revision=...)``.
            max_seq_length: Ignored — ColQwen3 uses dynamic sequence length.
            muvera_config: Optional MUVERA configuration (passed to postprocessor).
            token_dim: Per-token embedding dimension (320 for ColQwen3).
            max_num_visual_tokens: Cap on visual tokens per image (passed to processor).
        """
        self._model_name_or_path = str(model_name_or_path)
        self._normalize = normalize
        self._compute_precision = compute_precision
        self._trust_remote_code = trust_remote_code
        self._revision = revision
        self._max_num_visual_tokens = max_num_visual_tokens

        self._model: Any = None
        self._processor: Any = None
        # HF fast tokenizers are NOT thread-safe: applying per-call padding/truncation
        # reconfigures the underlying Rust tokenizer (a mutable borrow). The direct
        # adapter path (encode_pipeline.py: asyncio.to_thread, no per-model lock) lets
        # concurrent requests race with RuntimeError: Already borrowed (#2098). Serialise
        # the processor call — microseconds vs the GPU forward. Matches CLIP/SigLIP.
        self._tokenizer_lock = threading.Lock()
        # transformers' hidden-state recorder mutates model-layer forwards during
        # each call. Serialize both text and image forwards so concurrent requests
        # cannot race that patch/restore cycle.
        self._forward_lock = threading.Lock()
        self._device: str | None = None
        self._muvera_config = muvera_config
        self._multivector_dim: int = token_dim

    def load(self, device: str) -> None:
        """Load processor + model onto the specified device."""
        from transformers import AutoModel, AutoProcessor

        self._device = device

        dtype = self._resolve_dtype()
        attn_impl = self._resolve_attn_implementation(device)

        logger.info(
            "Loading ColQwen3 model %s on device=%s with dtype=%s, attn=%s",
            self._model_name_or_path,
            device,
            dtype,
            attn_impl,
        )

        self._processor = AutoProcessor.from_pretrained(
            self._model_name_or_path,
            trust_remote_code=self._trust_remote_code,
            max_num_visual_tokens=self._max_num_visual_tokens,
            revision=self._revision,
        )

        load_kwargs: dict[str, Any] = {
            "trust_remote_code": self._trust_remote_code,
            "device_map": device,
            "dtype": dtype,
        }
        if attn_impl is not None:
            load_kwargs["attn_implementation"] = attn_impl
        if self._revision is not None:
            load_kwargs["revision"] = self._revision

        self._model = AutoModel.from_pretrained(
            self._model_name_or_path,
            **load_kwargs,
        ).eval()

        # Discover token dim from the projection layer when present.
        proj = getattr(self._model, "embedding_proj_layer", None)
        out_features = getattr(proj, "out_features", None)
        if isinstance(out_features, int) and out_features > 0:
            self._multivector_dim = out_features

        # FIX[#1055]: propagate the requested attention impl onto the inner VLM.
        # The model builds ``self.vlm`` via ``AutoModelForImageTextToText.from_config``
        # with a *fresh* ``Qwen3VLConfig`` that does not inherit the
        # ``attn_implementation`` passed to ``from_pretrained`` (and its own
        # ``attn_impl`` constructor hook is never invoked by ``from_pretrained``),
        # so the inner VLM silently defaults to sdpa. On the sdpa/eager path
        # Qwen3-VL's vision attention falls back to a per-image Python loop over
        # ``torch.split`` (one attention call per image per layer), which
        # serializes the batch and starves the GPU (~27s/image, util ~31%).
        # Re-dispatching to flash_attention_2 routes the whole batch through the
        # single cu_seqlens flash-varlen call.
        self._apply_attn_implementation(attn_impl)

        # FIX[#1055]: replace the vision Conv3d patch-embed with its matmul
        # equivalent. It dominated the forward at ~169s/6-image batch (99.7% of
        # the vision tower; the transformer blocks were only ~140ms each) — see
        # rebind_vision_patch_embed for why.
        rebind_vision_patch_embed(self._model, "colqwen3")

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
        except ImportError:
            logger.info("flash_attn not available, using sdpa attention")
            return "sdpa"

    def _apply_attn_implementation(self, attn_impl: str | None) -> None:
        """Force the requested attention impl onto the inner Qwen3-VL (``self.vlm``).

        Needed because the requested impl does not propagate through the model's
        ``from_config`` construction. Best-effort: on failure we keep the inner
        default (sdpa) rather than break loading.
        """
        if attn_impl is None or attn_impl == "sdpa":
            return
        vlm = getattr(self._model, "vlm", None)
        if vlm is None or not hasattr(vlm, "set_attn_implementation"):
            logger.warning("ColQwen3: inner VLM cannot set attn_implementation; keeping default")
            return
        try:
            vlm.set_attn_implementation(attn_impl)
            logger.info("ColQwen3: set inner-VLM attn_implementation=%s", attn_impl)
        except Exception:  # noqa: BLE001 - best-effort, keep loaded default on failure
            logger.warning("ColQwen3: could not set inner-VLM attn=%s; keeping default", attn_impl, exc_info=True)

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
        self._validate_output_types(output_types)

        if is_query:
            multivector_list: list[np.ndarray] = []
            for item in items:
                if item.text is None:
                    raise ValueError(_ERR_NO_INPUT)
                multivector_list.append(self._encode_text(item.text))
            return EncodeOutput(
                multivector=multivector_list,
                batch_size=len(items),
                is_query=is_query,
                multivector_token_dim=self._multivector_dim,
            )

        # Preallocate by index so output order matches input order regardless of
        # text/image mix, and so multi-image items collapse to one multivector.
        results: list[np.ndarray | None] = [None] * len(items)
        all_images: list[PILImage.Image] = []
        image_slots: list[tuple[int, int]] = []  # (item_idx, image_count)
        for idx, item in enumerate(items):
            has_images = item.images is not None and len(item.images) > 0
            if has_images:
                images = self._load_images(item)
                all_images.extend(images)
                image_slots.append((idx, len(images)))
            elif item.text is not None:
                results[idx] = self._encode_text(item.text)
            else:
                raise ValueError(_ERR_NO_INPUT)

        if all_images:
            per_image_mvs = self._encode_images(all_images)
            cursor = 0
            for idx, count in image_slots:
                segment = per_image_mvs[cursor : cursor + count]
                cursor += count
                results[idx] = segment[0] if count == 1 else np.concatenate(segment, axis=0)

        multivector_list = [mv for mv in results if mv is not None]
        assert len(multivector_list) == len(items)

        return EncodeOutput(
            multivector=multivector_list,
            batch_size=len(items),
            is_query=is_query,
            multivector_token_dim=self._multivector_dim,
        )

    # ------------------------------------------------------------------
    # Image encoding
    # ------------------------------------------------------------------

    def _load_images(self, item: Any) -> list[PILImage.Image]:
        from PIL import Image

        pil_images: list[PILImage.Image] = []
        for img_input in item.images or []:
            pil_img = Image.open(io.BytesIO(media_bytes(img_input, kind="image")))
            if pil_img.mode != "RGB":
                pil_img = pil_img.convert("RGB")
            pil_images.append(pil_img)
        return pil_images

    def _encode_images(self, images: list[PILImage.Image]) -> list[np.ndarray]:
        """Encode a batch of images and return per-image multi-vectors."""
        assert self._model is not None
        assert self._processor is not None

        with self._tokenizer_lock:
            inputs = self._processor(
                images=images,
                return_tensors="pt",
                padding="longest",
            )
        inputs = {k: v.to(self._device) for k, v in inputs.items() if hasattr(v, "to")}

        with self._forward_lock, torch.inference_mode():
            outputs = self._model(**inputs)

        # ColQwen3 returns a ModelOutput-like object with ``.embeddings``
        # of shape (batch, seq, token_dim). The wrapper already L2-normalizes
        # and applies attention-masking; our ``self._normalize`` is a no-op
        # safety belt for downstream parity.
        embeddings = outputs.embeddings
        if self._normalize:
            embeddings = F.normalize(embeddings, p=2, dim=-1)

        results: list[np.ndarray] = [embeddings[i].float().cpu().numpy() for i in range(embeddings.shape[0])]

        # Free GPU memory between batches to prevent OOM on subsequent calls
        # (L4 22GB GPUs are tight for VLM models).
        del outputs, embeddings, inputs
        if self._device and self._device.startswith("cuda"):
            torch.cuda.empty_cache()

        return results

    # ------------------------------------------------------------------
    # Text encoding
    # ------------------------------------------------------------------

    def _encode_text(self, text: str) -> np.ndarray:
        """Encode a single text query."""
        assert self._model is not None
        assert self._processor is not None

        with self._tokenizer_lock:
            inputs = self._processor(
                text=[text],
                return_tensors="pt",
                padding="longest",
            )
        inputs = {k: v.to(self._device) for k, v in inputs.items() if hasattr(v, "to")}

        with self._forward_lock, torch.inference_mode():
            outputs = self._model(**inputs)

        embeddings = outputs.embeddings  # (1, seq, token_dim)
        if self._normalize:
            embeddings = F.normalize(embeddings, p=2, dim=-1)

        result = embeddings[0].float().cpu().numpy()

        del outputs, embeddings, inputs
        if self._device and self._device.startswith("cuda"):
            torch.cuda.empty_cache()

        return result

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
        self._check_loaded()

        query_output = self.encode([query], output_types=["multivector"], is_query=True)
        if query_output.multivector is None:
            raise RuntimeError("Failed to encode query: no multivector output")
        query_vecs = query_output.multivector[0]

        doc_output = self.encode(items, output_types=["multivector"], is_query=False)
        if doc_output.multivector is None:
            raise RuntimeError("Failed to encode documents: no multivector output")

        query_tensor = torch.from_numpy(query_vecs).to(self._device)
        doc_tensors = [torch.from_numpy(d).to(self._device) for d in doc_output.multivector]
        return maxsim_scores(query_tensor, doc_tensors)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _validate_output_types(self, output_types: list[str]) -> None:
        unsupported = set(output_types) - {"multivector"}
        if unsupported:
            msg = f"Unsupported output types: {unsupported}. ColQwen3Adapter only supports 'multivector'."
            raise ValueError(msg)

    def get_postprocessors(self) -> dict[str, Any]:
        """Return the configured MUVERA multivector-to-dense postprocessor."""
        config = MuveraConfig(**self._muvera_config) if self._muvera_config else MuveraConfig()
        return {"muvera": MuveraPostprocessor(token_dim=self._multivector_dim, config=config)}

    def get_preprocessor(self) -> Any | None:
        # ColQwen3 uses a custom processor that handles both text and images
        # internally via the ColQwen3Processor; the generic ImagePreprocessor
        # does not match the (text-only / image-only) call pattern.
        return None
