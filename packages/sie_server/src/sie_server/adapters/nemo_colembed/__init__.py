"""NeMo ColEmbed adapter for visual document retrieval.

This adapter supports NVIDIA's NeMo ColEmbed model for visual document retrieval.
The model encodes document images into multi-vector representations for late
interaction retrieval.

Target model: nvidia/llama-nemoretriever-colembed-3b-v1

Key features:
- Top performer on ViDoRe v3 benchmark
- Based on SigLIP2 + Llama architecture (3B params)
- Uses custom API: forward_queries() and forward_passages()
- Requires flash_attn and trust_remote_code=True

License: NVIDIA Non-Commercial (note in model config)

- Full conformance with SIE preprocessor infrastructure
- Uses NemoColEmbedPreprocessor for image preprocessing
- Calls model.forward() directly instead of forward_passages()
- Removes model-internal DataLoader (num_workers=8) overhead

See: https://huggingface.co/nvidia/llama-nemoretriever-colembed-3b-v1
"""

from __future__ import annotations

import io
import logging
import threading
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import torch

from sie_server.adapters._base_adapter import BaseAdapter
from sie_server.adapters._multivector import maxsim_scores
from sie_server.adapters._spec import AdapterSpec
from sie_server.adapters._types import ComputePrecision
from sie_server.adapters._vision_patch_embed import rebind_vision_patch_embed
from sie_server.core.inference_output import EncodeOutput
from sie_server.core.postprocessor import MuveraConfig, MuveraPostprocessor
from sie_server.types.inputs import media_bytes

if TYPE_CHECKING:
    from sie_server.types.inputs import Item

logger = logging.getLogger(__name__)

_ERR_NO_INPUT = "NemoColEmbedAdapter requires either text or images input"
_ERR_REQUIRES_FLASH_ATTN = (
    "NemoColEmbedAdapter requires flash_attn. Install with: pip install flash-attn --no-build-isolation"
)

# Serializes v1 dynamic-class resolution and the ``config_class.to_dict``
# monkey-patch so concurrent loads cannot observe a temporary serializer.
_V1_TO_DICT_PATCH_LOCK = threading.Lock()


class NemoColEmbedAdapter(BaseAdapter):
    """Adapter for NVIDIA NeMo ColEmbed visual document retrieval model.

    NeMo ColEmbed encodes document page images into multi-vector representations
    for late interaction retrieval. Top performer on ViDoRe v3 benchmark.

    Uses custom model API:
    - forward_queries(texts) for text query encoding
    - forward_passages(images) for document image encoding
    - get_scores() for MaxSim scoring

    Requires flash_attn and trust_remote_code=True.
    """

    spec = AdapterSpec(
        inputs=("text", "image"),
        outputs=("multivector", "score"),
        # v1 emits the Llama hidden width (3072); v2 emits its Qwen3-VL
        # hidden width (2560). Keep this instance-derived so one shared
        # adapter contract does not mask either model's configured token_dim.
        multivector_dim=None,
        unload_fields=("_model", "_processor"),
        default_preprocessor="charcount",
    )

    def __init__(
        self,
        model_name_or_path: str | Path,
        *,
        normalize: bool = True,
        compute_precision: ComputePrecision = "bfloat16",
        revision: str | None = None,
        max_seq_length: int | None = None,
        batch_size: int = 8,
        muvera_config: dict[str, Any] | None = None,
        token_dim: int = 3072,
    ) -> None:
        """Initialize the adapter.

        Args:
            model_name_or_path: HuggingFace model ID or local path.
            normalize: Whether to L2-normalize embeddings.
            compute_precision: Compute precision for inference.
            revision: Optional HuggingFace revision/branch/commit SHA to pin when
                loading model artifacts. Forwarded to ``from_pretrained(..., revision=...)``.
            max_seq_length: Ignored - model uses dynamic sequence length.
            batch_size: Batch size for encoding (passed to model methods).
            muvera_config: MUVERA configuration (passed to postprocessor, not used by adapter).
            token_dim: Token embedding dimension used until the loaded model
                configuration provides its effective hidden width.
        """
        self._model_name_or_path = str(model_name_or_path)
        self._normalize = normalize
        self._compute_precision = compute_precision
        self._revision = revision
        self._batch_size = batch_size

        self._model: Any = None
        # The v1 wrappers and direct prepared path request hidden states. If the
        # remote model uses transformers' process-global output recorder, its
        # unlocked layer-forward patch/restore leaks GPU memory under interleaved
        # forwards. Keep every model call behind one boundary (#2144/#2204).
        self._forward_lock = threading.Lock()
        self._device: str | None = None
        self._muvera_config = muvera_config
        self._multivector_dim: int = token_dim
        self._processor: Any = None  # NemoColEmbedPreprocessor, created on load()
        # Note: Named _processor (not _preprocessor) for PreprocessorRegistry auto-detection

    def load(self, device: str) -> None:
        """Load the model onto the specified device.

        Args:
            device: Device string (e.g., "cuda:0", "cpu").

        Raises:
            ImportError: If flash_attn is required by the model and not installed.
        """
        from transformers import AutoModel

        self._device = device

        # Determine dtype
        dtype = self._resolve_dtype()
        attn_impl = self._resolve_attn_implementation(device)

        logger.info(
            "Loading NeMo ColEmbed model %s on device=%s with dtype=%s, attn=%s",
            self._model_name_or_path,
            device,
            dtype,
            attn_impl,
        )

        # Try AutoModel first (works for v2: Qwen3-VL backbone, registers cleanly).
        # Fall back to v1's explicit dynamic-module path with the to_dict monkey-patch
        # if AutoModel can't resolve the custom architecture. The fallback covers:
        #  - KeyError / ValueError: AutoModel can't find a matching architecture
        #  - ImportError: the dynamic modeling file imports flash_attn at module
        #    top (v1 ships its own modeling file that does this)
        #  - AttributeError: v1's config.to_dict() bug fires inside transformers'
        #    to_diff_dict() comparison when vision_config is absent
        automodel_kwargs: dict[str, Any] = {
            "trust_remote_code": True,
            "device_map": device,
            "torch_dtype": dtype,
            "attn_implementation": attn_impl,
        }
        if self._revision is not None:
            automodel_kwargs["revision"] = self._revision
        loaded_via_v1_path = False
        try:
            self._model = AutoModel.from_pretrained(
                self._model_name_or_path,
                **automodel_kwargs,
            )
        except (KeyError, ValueError, ImportError, AttributeError) as automodel_err:
            logger.info(
                "AutoModel.from_pretrained failed for %s (%s: %s); falling back to v1 dynamic-module load path",
                self._model_name_or_path,
                type(automodel_err).__name__,
                automodel_err,
            )
            self._model = self._load_v1_dynamic(device, dtype)
            loaded_via_v1_path = True

        self._model.eval()

        # v2 exposes its 2560-d token width under ``text_config``. v1 returns
        # ``hidden_states[-1]`` from the nested Llama backbone, so its 3072-d
        # output width lives under ``config.llm_config.hidden_size``.
        # The stale 128-d fallback used to build an 8,192-wide MUVERA
        # repetition block for a 196,608-wide result (64 * 3072).
        embedding_dim = self._discover_multivector_dim(self._model.config)
        if embedding_dim is not None:
            self._multivector_dim = embedding_dim

        # FIX[#1055]: ensure the requested attention impl actually reaches the
        # inner Qwen3-VL backbone (v2). On the sdpa fallback Qwen3-VL's vision
        # attention runs a per-image Python loop over torch.split that
        # serializes the batch (see ColQwen3). Safe no-op when flash already
        # propagated, or for the v1 SigLIP+Llama path.
        if attn_impl is not None and attn_impl != "sdpa" and hasattr(self._model, "set_attn_implementation"):
            try:
                self._model.set_attn_implementation(attn_impl)
            except Exception:  # noqa: BLE001 - best-effort, keep loaded default on failure
                logger.warning("NemoColEmbed: could not set attn=%s; keeping default", attn_impl, exc_info=True)

        # v1 (SigLIP+Llama) vs v2 (Qwen3-VL) backbone. v1 exposes image_size /
        # num_image_token (and always takes the dynamic-module fallback path); v2
        # does not.
        is_v1 = loaded_via_v1_path or (hasattr(self._model, "image_size") and hasattr(self._model, "num_image_token"))

        # FIX[#1055]/[#1151]: rebind the Qwen3-VL (v2) vision Conv3d patch-embed to
        # its matmul equivalent (same weights). The non-overlapping Conv3d hits a
        # pathologically slow per-patch cuDNN path that dominated the vision-tower
        # forward (~99.7% for the sibling ColQwen3). Restrict to v2: the v1 SigLIP
        # patch conv runs over the full image (not pre-patched rows), so the
        # per-patch F.linear rebind would mis-shape it — and the load-time probe
        # (per-patch shaped) would not catch that.
        if not is_v1:
            rebind_vision_patch_embed(self._model, "nemo_colembed")

        # Create preprocessor only for v1 — the SigLIP+Llama-keyed
        # NemoColEmbedPreprocessor does not apply to the v2 (Qwen3-VL) backbone.
        if is_v1:
            self._create_processor()
        else:
            logger.info("Skipping NemoColEmbedPreprocessor — model lacks v1 attrs (likely v2 backbone).")

    @staticmethod
    def _discover_multivector_dim(config: Any) -> int | None:
        """Return the token width emitted by the loaded ColEmbed model."""
        for candidate in (
            getattr(config, "hidden_size", None),
            getattr(getattr(config, "text_config", None), "hidden_size", None),
            getattr(getattr(config, "llm_config", None), "hidden_size", None),
            getattr(config, "embedding_dim", None),
        ):
            if isinstance(candidate, int) and candidate > 0:
                return candidate
        return None

    def _load_v1_dynamic(self, device: str, dtype: torch.dtype) -> Any:
        """Load v1 (SigLIP+Llama) via the explicit dynamic-module path.

        This path applies a ``to_dict`` monkey-patch to work around a v1
        config bug that assumes ``vision_config`` always exists. v1 also
        requires ``flash_attention_2``.
        """
        try:
            import flash_attn  # ty: ignore[unresolved-import]
        except ImportError as e:
            raise ImportError(_ERR_REQUIRES_FLASH_ATTN) from e

        from transformers.dynamic_module_utils import get_class_from_dynamic_module

        # v1's config.to_dict() bug: assumes vision_config exists but it doesn't.
        # Triggered during transformers' to_diff_dict() comparison. Patch around it
        # only for this v1 fallback path — v2 inherits from Qwen3VLConfig cleanly.
        def patched_to_dict(self: Any) -> dict[str, Any]:
            """Patched to_dict that handles missing vision_config (v1 only)."""
            import copy

            output = copy.deepcopy(self.__dict__)
            if hasattr(self, "vision_config") and self.vision_config is not None:
                output["vision_config"] = self.vision_config.to_dict()
            if hasattr(self, "llm_config") and self.llm_config is not None:
                output["llm_config"] = self.llm_config.to_dict()
            output["model_type"] = self.__class__.model_type
            return output

        v1_kwargs: dict[str, Any] = {
            "device_map": device,
            "torch_dtype": dtype,
            "attn_implementation": "flash_attention_2",
        }
        if self._revision is not None:
            v1_kwargs["revision"] = self._revision

        # Dynamic-module resolution can return the same cached class while
        # another load has its config serializer patched. Serialize resolution,
        # patching, loading, and restoration as one critical section so no
        # concurrent v1 load can observe or reuse the temporary serializer.
        with _V1_TO_DICT_PATCH_LOCK:
            model_class = get_class_from_dynamic_module(
                "modeling_llama_nemoretrievercolembed.llama_NemoRetrieverColEmbed",
                self._model_name_or_path,
                trust_remote_code=True,
                revision=self._revision,
            )
            config_class = model_class.config_class  # ty:ignore[unresolved-attribute]
            original_to_dict = config_class.to_dict
            config_class.to_dict = patched_to_dict
            try:
                return model_class.from_pretrained(  # ty:ignore[unresolved-attribute]
                    self._model_name_or_path,
                    **v1_kwargs,
                )
            finally:
                config_class.to_dict = original_to_dict

    def _resolve_attn_implementation(self, device: str) -> str:
        """Return ``flash_attention_2`` when available on CUDA, else ``sdpa``.

        v2 (Qwen3-VL) tolerates ``sdpa`` cleanly. v1 still requires
        ``flash_attention_2`` and that requirement is enforced inside
        ``_load_v1_dynamic`` via an explicit ``import flash_attn`` check.
        """
        if not str(device).startswith("cuda"):
            return "sdpa"
        try:
            import flash_attn  # ty: ignore[unresolved-import]

            return "flash_attention_2"
        except ImportError:
            logger.info("flash_attn not available, using sdpa attention")
            return "sdpa"

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
        return dtype_map.get(self._compute_precision, torch.bfloat16)

    def _create_processor(self) -> None:
        """Create preprocessor for image preprocessing.

        Uses model's tokenizer and config for full conformance.
        This enables SIE's thread pool and batching infrastructure
        instead of the model's internal DataLoader.

        Note: Named _processor for PreprocessorRegistry auto-detection.
        """
        from sie_server.core.preprocessor import NemoColEmbedPreprocessor

        # Get preprocessing parameters from model
        image_size = getattr(self._model, "image_size", 448)
        max_input_tiles = getattr(self._model, "max_input_tiles", 6)
        use_thumbnail = getattr(self._model.config, "use_thumbnail", False)
        num_image_token = getattr(self._model, "num_image_token", 256)

        self._processor = NemoColEmbedPreprocessor(
            tokenizer=self._model.tokenizer,
            model_config=self._model.config,
            model_name=self._model_name_or_path,
            image_size=image_size,
            max_input_tiles=max_input_tiles,
            use_thumbnail=use_thumbnail,
            num_image_token=num_image_token,
        )

        logger.info(
            "Created NemoColEmbedPreprocessor: image_size=%d, max_tiles=%d, num_image_token=%d",
            image_size,
            max_input_tiles,
            num_image_token,
        )

    def get_preprocessor(self) -> Any:
        """Register BOTH a text and an image preprocessor for v1 (#1163).

        v1 documents must take the conformant ``_encode_images_preprocessed`` path,
        which requires an *image* preprocessor to be registered so the encode pipeline
        produces a ``NemoColEmbedPayload`` (with ``pixel_values``) instead of a
        passthrough ``ImagePayload``. Without it every doc batch falls back to the
        model's ``forward_passages`` — which re-tiles each page inline on one thread,
        ~3x slower than running the tiling upstream in the preprocessing thread pool.

        But v1 *queries* (text) still go through ``model.forward_queries`` and rely on
        the batched worker path; registering only the image preprocessor de-registers
        the text one and routes queries to the unbatched direct-call path, which has
        surfaced ``forward_queries`` failures. So we register both: the base
        ``CharCountPreprocessor`` (text → worker-batched queries) and the
        ``NemoColEmbedPreprocessor`` (image → conformant docs). ``model_loader``
        registers each entry of the returned list by its ``modality``.

        v2 (Qwen3-VL backbone) builds no ``_processor`` (``None``); it keeps just the
        base text preprocessor and its native ``forward_images`` path (with #1055 fix).
        """
        if self._processor is None:
            return super().get_preprocessor()
        return [super().get_preprocessor(), self._processor]

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

        For document images: uses forward() directly with preprocessed inputs,
        or falls back to forward_passages() if no prepared_items provided.
        For text queries: uses forward_queries() returning per-token embeddings.

        Args:
            items: List of items to encode (with text or images).
            output_types: Which outputs to return (only "multivector" supported).
            instruction: Optional instruction (not used).
            is_query: Whether items are queries (True) or documents (False).
                For queries, expects text input.
                For documents, expects image input.
            prepared_items: Preprocessed items from NemoColEmbedPreprocessor.
                If provided for documents, calls model.forward() directly.

        Returns:
            EncodeOutput with multivector embeddings.
        """
        self._check_loaded()

        self._validate_output_types(output_types)

        # Batch encode based on query vs document
        if is_query:
            return self._encode_texts(items, is_query=is_query)

        # Use preprocessed inputs if available (conformant path)
        # Only use if prepared_items have NemoColEmbedPayload (not dummy ImagePayload)
        if prepared_items is not None and len(prepared_items) > 0:
            from sie_server.core.prepared import NemoColEmbedPayload, PreparedItem

            # Check first item to see if it has valid NemoColEmbedPayload
            first = prepared_items[0]
            if isinstance(first, PreparedItem):
                payload = first.payload
            else:
                payload = getattr(first, "payload", None)

            if isinstance(payload, NemoColEmbedPayload) and payload.pixel_values is not None:
                return self._encode_images_preprocessed(items, prepared_items, is_query=is_query)

        # Fallback: use model's internal preprocessing (legacy path)
        return self._encode_images(items, is_query=is_query)

    def _encode_texts(self, items: list[Any], *, is_query: bool) -> EncodeOutput:
        """Encode text queries using forward_queries().

        Args:
            items: List of items with text.
            is_query: Whether items are queries.

        Returns:
            EncodeOutput with multivector embeddings.
        """
        from torch.nn import functional

        # Item is a TypedDict (dict) - no instance check needed
        texts = []
        for item in items:
            if item.text is None:
                raise ValueError(_ERR_NO_INPUT)
            texts.append(item.text)

        # Use model's forward_queries method
        with self._forward_lock, torch.inference_mode():
            embeddings = self._model.forward_queries(texts, batch_size=self._batch_size)

        multivector_list = self._unpack_embeddings(embeddings, expected_count=len(texts))

        # Free GPU memory to prevent OOM on subsequent calls
        del embeddings
        if self._device and self._device.startswith("cuda"):
            torch.cuda.empty_cache()
        _ = functional  # silence unused-import in some lint passes

        return EncodeOutput(
            multivector=multivector_list,
            batch_size=len(items),
            is_query=is_query,
            multivector_token_dim=self._multivector_dim,
        )

    def _encode_images(self, items: list[Any], *, is_query: bool) -> EncodeOutput:
        """Encode document images using forward_images() (v2) or forward_passages() (v1).

        Args:
            items: List of items with images.
            is_query: Whether items are queries.

        Returns:
            EncodeOutput with multivector embeddings.
        """
        from PIL import Image

        # Item is a TypedDict (dict) - no instance check needed
        pil_images = []
        for item in items:
            if not item.images or len(item.images) == 0:
                raise ValueError(_ERR_NO_INPUT)

            # Load first image from each item (ImageInput is also a TypedDict)
            img_bytes = media_bytes(item.images[0], kind="image")
            pil_img = Image.open(io.BytesIO(img_bytes))
            if pil_img.mode != "RGB":
                pil_img = pil_img.convert("RGB")
            pil_images.append(pil_img)

        # v2 exposes ``forward_images`` (Qwen3-VL backbone); v1 exposes ``forward_passages``.
        forward = getattr(self._model, "forward_images", None) or self._model.forward_passages

        with self._forward_lock, torch.inference_mode():
            embeddings = forward(pil_images, batch_size=self._batch_size)

        multivector_list = self._unpack_embeddings(embeddings, expected_count=len(pil_images))

        # Free GPU memory to prevent OOM on subsequent calls
        del embeddings
        if self._device and self._device.startswith("cuda"):
            torch.cuda.empty_cache()

        return EncodeOutput(
            multivector=multivector_list,
            batch_size=len(items),
            is_query=is_query,
            multivector_token_dim=self._multivector_dim,
        )

    def _unpack_embeddings(
        self,
        embeddings: Any,
        *,
        expected_count: int,
    ) -> list[np.ndarray]:
        """Convert v1's list-of-Tensor or v2's padded (B, S, D) tensor to per-item numpy.

        v1 ``forward_queries``/``forward_passages`` return ``list[Tensor]`` where
        each entry is ``[seq, dim]``. v2 returns a single padded tensor of shape
        ``[batch, max_seq, dim]``. Normalize per-row when ``self._normalize``.
        """
        from torch.nn import functional

        results: list[np.ndarray] = []

        if isinstance(embeddings, torch.Tensor):
            if embeddings.ndim != 3:
                msg = (
                    f"NemoColEmbed: unexpected embedding tensor shape {tuple(embeddings.shape)}; "
                    f"expected 3D [batch, seq, dim]"
                )
                raise RuntimeError(msg)
            # v2 right-pads with zero rows so all rows in a batch share a sequence
            # length. Detect padding before normalization (zero rows become NaN
            # after L2 normalize) and drop them per-item so downstream MaxSim
            # doesn't consume padding tokens as content.
            nonzero_mask = embeddings.abs().sum(dim=-1) > 0
            tensor = embeddings
            if self._normalize:
                tensor = functional.normalize(tensor, p=2, dim=-1)
            for i in range(tensor.shape[0]):
                keep = nonzero_mask[i]
                results.append(tensor[i][keep].float().cpu().numpy())
            return results

        # Iterable of per-item tensors / arrays (v1 path)
        for emb in embeddings:
            if isinstance(emb, torch.Tensor):
                if self._normalize:
                    emb = functional.normalize(emb, p=2, dim=-1)
                results.append(emb.float().cpu().numpy())
            else:
                arr = np.asarray(emb, dtype=np.float32)
                if self._normalize:
                    arr = arr / np.linalg.norm(arr, axis=-1, keepdims=True)
                results.append(arr)

        if len(results) != expected_count:
            logger.warning("NemoColEmbed: expected %d embeddings, got %d", expected_count, len(results))
        return results

    def _encode_images_preprocessed(
        self,
        items: list[Any],
        prepared_items: list[Any],
        *,
        is_query: bool,
    ) -> EncodeOutput:
        """Encode document images using preprocessed inputs.

        Calls model.forward() directly instead of forward_passages(),
        bypassing the model's internal DataLoader preprocessing.

        Sub-batches items according to self._batch_size to avoid OOM
        on large batches (similar to how forward_passages() handles it).

        Args:
            items: Original items (for extracting item IDs).
            prepared_items: Preprocessed items from NemoColEmbedPreprocessor.
            is_query: Whether items are queries.

        Returns:
            EncodeOutput with multivector embeddings.
        """
        from torch.nn import functional

        from sie_server.types.inputs import Item

        # Belt-and-braces: if no v1 preprocessor (e.g., v2 backbone), defer to the
        # native forward_images / forward_passages path. The dispatch in ``encode()``
        # already short-circuits on payload type, so this only triggers when a caller
        # invokes _encode_images_preprocessed directly.
        if self._processor is None:
            return self._encode_images(items, is_query=is_query)

        # Process in sub-batches to maximize GPU utilization while avoiding OOM
        # NemoColEmbed supports batching - model maps tiles to sequences via IMG_CONTEXT token counts
        all_embeddings: list[np.ndarray] = []
        batch_size = self._batch_size  # Default 8, same as model's forward_passages()

        for start_idx in range(0, len(prepared_items), batch_size):
            end_idx = min(start_idx + batch_size, len(prepared_items))
            sub_batch_items = prepared_items[start_idx:end_idx]

            try:
                # Collate sub-batch - concatenates pixel_values, pads input_ids
                batch = self._processor.collate(sub_batch_items, device=self._device)

                # Forward pass - call model directly
                with (
                    self._forward_lock,
                    torch.inference_mode(),
                    torch.autocast(device_type="cuda", dtype=torch.bfloat16),
                ):
                    outputs = self._model(
                        pixel_values=batch["pixel_values"],
                        input_ids=batch["input_ids"],
                        attention_mask=batch["attention_mask"],
                        output_hidden_states=True,
                    )
            except Exception:
                logger.exception("NemoColEmbed forward failed for batch %d-%d", start_idx, end_idx)
                raise

            # Extract embeddings from last hidden state: [batch_size, seq_len, hidden_dim]
            embeddings = outputs.hidden_states[-1]

            # Mask padding tokens (multiply by attention mask)
            attention_mask = batch["attention_mask"].unsqueeze(-1)
            embeddings = embeddings * attention_mask

            # Normalize
            if self._normalize:
                embeddings = functional.normalize(embeddings, p=2, dim=-1)

            # Store results for this sub-batch (move to CPU immediately to free GPU
            # memory). Trim each item's left-padding rows before returning: the batch is
            # left-padded, so padded positions are zeroed by the attention_mask above —
            # but emitting them as zero vectors leaks 0-similarity rows into the late-
            # interaction MaxSim (a 0-floor on every query token's max). Because the
            # batcher pads inconsistently across docs, identical docs then score
            # differently by batch and ranking is corrupted on variable-tile batches
            # (#1163: Vidore3 Hr 0.6532 -> 0.5713). Keep only real tokens, matching the
            # native forward_passages path (_unpack_embeddings drops zero rows likewise).
            for i in range(len(sub_batch_items)):
                keep = batch["attention_mask"][i].bool()
                emb = embeddings[i][keep].float().cpu().numpy()
                all_embeddings.append(emb)

            # Free this sub-batch's GPU tensors. NOTE: no per-sub-batch
            # torch.cuda.empty_cache() — repeatedly releasing the allocator's cache and
            # re-acquiring ~GB blocks fragments the pool and OOMs at scale on big GPUs
            # (#1163). The sub-batch loop + immediate CPU offload already bound peak VRAM.
            del outputs, embeddings, attention_mask, batch

        return EncodeOutput(
            multivector=all_embeddings,
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
        """Score document images against a text query.

        Uses the model's built-in get_scores() method for efficient scoring.

        Args:
            query: Query item (with text).
            items: List of document items (with images).
            instruction: Optional instruction (not used).
            options: Optional options (not used).

        Returns:
            List of scores, one per document.
        """
        self._check_loaded()

        # Encode query and documents
        query_multivector = self.encode([query], output_types=["multivector"], is_query=True).multivector
        doc_multivector = self.encode(items, output_types=["multivector"], is_query=False).multivector

        if query_multivector is None:
            raise RuntimeError("Query encoding returned None multivector embeddings")
        if doc_multivector is None:
            raise RuntimeError("Document encoding returned None multivector embeddings")

        # Convert to tensors for get_scores
        query_emb = torch.from_numpy(query_multivector[0]).to(self._device)
        doc_embs = [torch.from_numpy(emb).to(self._device) for emb in doc_multivector]

        # Use model's get_scores if available, otherwise compute MaxSim manually
        if hasattr(self._model, "get_scores"):
            # Model returns [num_queries, num_docs] similarity matrix
            scores_matrix = self._model.get_scores([query_emb], doc_embs)
            return scores_matrix[0].cpu().tolist()

        # Fallback: compute MaxSim manually.
        return maxsim_scores(query_emb, doc_embs)

    def _validate_output_types(self, output_types: list[str]) -> None:
        """Validate that output types are supported."""
        unsupported = set(output_types) - {"multivector"}
        if unsupported:
            msg = f"Unsupported output types: {unsupported}. NemoColEmbedAdapter only supports 'multivector'."
            raise ValueError(msg)

    def get_postprocessors(self) -> dict[str, Any]:
        """Return the configured MUVERA multivector-to-dense postprocessor."""
        config = MuveraConfig(**self._muvera_config) if self._muvera_config else MuveraConfig()
        return {"muvera": MuveraPostprocessor(token_dim=self._multivector_dim, config=config)}
