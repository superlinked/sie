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

Per roadmap Project 10.5 Phase 1c:
- Full conformance with SIE preprocessor infrastructure
- Uses NemoColEmbedPreprocessor for image preprocessing
- Calls model.forward() directly instead of forward_passages()
- Removes model-internal DataLoader (num_workers=8) overhead

See: https://huggingface.co/nvidia/llama-nemoretriever-colembed-3b-v1
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
from sie_server.core.preprocessor import CharCountPreprocessor

if TYPE_CHECKING:
    from sie_server.types.inputs import Item

logger = logging.getLogger(__name__)

ComputePrecision = Literal["float16", "bfloat16", "float32"]

_ERR_NOT_LOADED = "Model not loaded. Call load() first."
_ERR_NO_INPUT = "NemoColEmbedAdapter requires either text or images input"
_ERR_REQUIRES_FLASH_ATTN = (
    "NemoColEmbedAdapter requires flash_attn. Install with: pip install flash-attn --no-build-isolation"
)


class NemoColEmbedAdapter(ModelAdapter):
    """Adapter for NVIDIA NeMo ColEmbed visual document retrieval model.

    NeMo ColEmbed encodes document page images into multi-vector representations
    for late interaction retrieval. Top performer on ViDoRe v3 benchmark.

    Uses custom model API:
    - forward_queries(texts) for text query encoding
    - forward_passages(images) for document image encoding
    - get_scores() for MaxSim scoring

    Requires flash_attn and trust_remote_code=True.
    """

    def __init__(
        self,
        model_name_or_path: str | Path,
        *,
        normalize: bool = True,
        compute_precision: ComputePrecision = "bfloat16",
        max_seq_length: int | None = None,
        batch_size: int = 8,
        muvera_config: dict[str, Any] | None = None,
        token_dim: int = 128,
    ) -> None:
        """Initialize the adapter.

        Args:
            model_name_or_path: HuggingFace model ID or local path.
            normalize: Whether to L2-normalize embeddings.
            compute_precision: Compute precision for inference.
            max_seq_length: Ignored - model uses dynamic sequence length.
            batch_size: Batch size for encoding (passed to model methods).
            muvera_config: MUVERA configuration (passed to postprocessor, not used by adapter).
            token_dim: Token embedding dimension (stored but not used, model has fixed 128-dim).
        """
        self._model_name_or_path = str(model_name_or_path)
        self._normalize = normalize
        self._compute_precision = compute_precision
        self._batch_size = batch_size

        self._model: Any = None
        self._device: str | None = None
        self._multivector_dim: int = token_dim  # ColEmbed uses 128-dim per patch
        self._processor: Any = None  # NemoColEmbedPreprocessor, created on load()
        # Note: Named _processor (not _preprocessor) for PreprocessorRegistry auto-detection

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

        Raises:
            ImportError: If flash_attn is not installed.
        """
        # Check flash_attn is available (required by this model)
        try:
            import flash_attn  # ty: ignore[unresolved-import]
        except ImportError as e:
            raise ImportError(_ERR_REQUIRES_FLASH_ATTN) from e

        from transformers.dynamic_module_utils import get_class_from_dynamic_module

        self._device = device

        # Determine dtype
        dtype = self._resolve_dtype(device)

        logger.info(
            "Loading NeMo ColEmbed model %s on device=%s with dtype=%s",
            self._model_name_or_path,
            device,
            dtype,
        )

        # Load model class directly to bypass AutoModel/AutoConfig introspection.
        model_class = get_class_from_dynamic_module(
            "modeling_llama_nemoretrievercolembed.llama_NemoRetrieverColEmbed",
            self._model_name_or_path,
            trust_remote_code=True,
        )

        # NVIDIA's config has a bug: to_dict() assumes vision_config exists but it doesn't.
        # The bug is triggered during transformers' to_diff_dict() which creates a new config
        # instance to compare defaults. Monkey-patch the config class to fix it.
        config_class = model_class.config_class  # ty:ignore[unresolved-attribute]
        original_to_dict = config_class.to_dict

        def patched_to_dict(self: Any) -> dict[str, Any]:
            """Patched to_dict that handles missing vision_config."""
            import copy

            output = copy.deepcopy(self.__dict__)
            # Only include vision_config if it exists
            if hasattr(self, "vision_config") and self.vision_config is not None:
                output["vision_config"] = self.vision_config.to_dict()
            # Only include llm_config if it exists
            if hasattr(self, "llm_config") and self.llm_config is not None:
                output["llm_config"] = self.llm_config.to_dict()
            output["model_type"] = self.__class__.model_type
            return output

        config_class.to_dict = patched_to_dict

        try:
            self._model = model_class.from_pretrained(  # ty:ignore[unresolved-attribute]
                self._model_name_or_path,
                device_map=device,
                torch_dtype=dtype,
                attn_implementation="flash_attention_2",
            )
        finally:
            # Restore original to_dict after loading
            config_class.to_dict = original_to_dict

        self._model.eval()

        # Get embedding dimension from model config if available
        if hasattr(self._model.config, "embedding_dim"):
            self._multivector_dim = self._model.config.embedding_dim

        # Create preprocessor for conformance with SIE infrastructure
        self._create_processor()

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
        if self._model is None:
            raise RuntimeError(_ERR_NOT_LOADED)

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
        with torch.inference_mode():
            embeddings = self._model.forward_queries(texts, batch_size=self._batch_size)

        # embeddings is a list of tensors [seq_len, dim] for each query
        multivector_list = []
        for emb in embeddings:
            if isinstance(emb, torch.Tensor):
                if self._normalize:
                    emb = functional.normalize(emb, p=2, dim=-1)
                emb = emb.float().cpu().numpy()
            else:
                emb = np.array(emb, dtype=np.float32)
                if self._normalize:
                    emb = emb / np.linalg.norm(emb, axis=-1, keepdims=True)

            multivector_list.append(emb)

        return EncodeOutput(
            multivector=multivector_list,
            batch_size=len(items),
            is_query=is_query,
            multivector_token_dim=self._multivector_dim,
        )

    def _encode_images(self, items: list[Any], *, is_query: bool) -> EncodeOutput:
        """Encode document images using forward_passages().

        Args:
            items: List of items with images.
            is_query: Whether items are queries.

        Returns:
            EncodeOutput with multivector embeddings.
        """
        from PIL import Image
        from torch.nn import functional

        # Item is a TypedDict (dict) - no instance check needed
        pil_images = []
        for item in items:
            if not item.images or len(item.images) == 0:
                raise ValueError(_ERR_NO_INPUT)

            # Load first image from each item (ImageInput is also a TypedDict)
            img_bytes = item.images[0]["data"]
            pil_img = Image.open(io.BytesIO(img_bytes))
            if pil_img.mode != "RGB":
                pil_img = pil_img.convert("RGB")
            pil_images.append(pil_img)

        # Use model's forward_passages method
        with torch.inference_mode():
            embeddings = self._model.forward_passages(pil_images, batch_size=self._batch_size)

        # embeddings is a list of tensors [num_patches, dim] for each image
        multivector_list = []
        for emb in embeddings:
            if isinstance(emb, torch.Tensor):
                if self._normalize:
                    emb = functional.normalize(emb, p=2, dim=-1)
                emb = emb.float().cpu().numpy()
            else:
                emb = np.array(emb, dtype=np.float32)
                if self._normalize:
                    emb = emb / np.linalg.norm(emb, axis=-1, keepdims=True)

            multivector_list.append(emb)

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
                with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
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

            # Store results for this sub-batch (move to CPU immediately to free GPU memory)
            for i in range(len(sub_batch_items)):
                emb = embeddings[i].float().cpu().numpy()
                all_embeddings.append(emb)

            # Clear GPU memory between sub-batches
            del outputs, embeddings, batch
            torch.cuda.empty_cache()

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
        if self._model is None:
            raise RuntimeError(_ERR_NOT_LOADED)

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

        # Fallback: compute MaxSim manually
        scores = []
        for doc_emb in doc_embs:
            sim = torch.matmul(query_emb, doc_emb.T)
            maxsim_score = sim.max(dim=-1).values.sum().item()
            scores.append(maxsim_score)
        return scores

    def _validate_output_types(self, output_types: list[str]) -> None:
        """Validate that output types are supported."""
        unsupported = set(output_types) - {"multivector"}
        if unsupported:
            msg = f"Unsupported output types: {unsupported}. NemoColEmbedAdapter only supports 'multivector'."
            raise ValueError(msg)

    def get_preprocessor(self) -> CharCountPreprocessor:
        """Return CharCountPreprocessor for cost estimation without tokenization overhead."""
        return CharCountPreprocessor(model_name=self._model_name_or_path)
