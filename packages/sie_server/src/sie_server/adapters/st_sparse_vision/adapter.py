"""Sparse-encoder adapter for vision-capable SparseEncoder checkpoints.

Serves sentence-transformers ``SparseEncoder`` models that accept image
documents in addition to text (visual-SPLADE family, e.g.
naver/v-splade-quality): text or page-image in, sparse vocab-sized vector out.

Requires the ``transformers5`` bundle (sentence-transformers >=5.6 /
transformers >=5.14); the multimodal ``SparseEncoder`` routing does not exist
on the sentence-transformers 5.0 line the default bundle pins.
"""

from __future__ import annotations

import io
import sys
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
from huggingface_hub import snapshot_download
from PIL import Image
from sentence_transformers import SparseEncoder

from sie_server.adapters._base_adapter import BaseAdapter
from sie_server.adapters._spec import AdapterSpec
from sie_server.adapters._types import ERR_NOT_LOADED, ComputePrecision
from sie_server.adapters._utils import validate_output_types
from sie_server.core.inference_output import EncodeOutput, SparseVector
from sie_server.core.loader import is_immutable_revision
from sie_server.types.inputs import Item, media_bytes

ERR_REQUIRES_TEXT_OR_IMAGE = "SparseEncoderVisionAdapter requires text or image input per item"


class SparseEncoderVisionAdapter(BaseAdapter):
    """Adapter for sparse sentence-transformers models with image input.

    Uses the SparseEncoder class for visual-SPLADE style models that encode
    text queries and document page images into sparse (vocab-dimension,
    mostly-zero) vectors scored by dot product.
    """

    spec = AdapterSpec(
        inputs=("text", "image"),
        outputs=("sparse",),
        unload_fields=("_model", "_sparse_dim"),
    )

    def __init__(
        self,
        model_name_or_path: str | Path,
        *,
        trust_remote_code: bool = True,
        max_seq_length: int | None = None,
        compute_precision: ComputePrecision = "float32",
        revision: str | None = None,
        **kwargs: Any,  # Accept extra args from loader (e.g., pooling)
    ) -> None:
        """Initialize the adapter.

        Args:
            model_name_or_path: HuggingFace model ID or local path.
            trust_remote_code: Whether to trust remote code in model files.
            max_seq_length: Override default max sequence length.
            compute_precision: Weight dtype passed to the checkpoint load.
                Defaults to float32: sparse retrieval scores are tie-dense and
                precision-sensitive (#1536), and the vanilla reference recipe
                for this family measures at float32.
            revision: Optional HuggingFace revision/branch/commit SHA to pin
                when loading model artifacts.
            **kwargs: Additional arguments (ignored, for compatibility).
        """
        _ = kwargs  # Unused, but accepted for loader compatibility
        self._model_name_or_path = str(model_name_or_path)
        self._trust_remote_code = trust_remote_code
        self._max_seq_length = max_seq_length
        self._compute_precision = compute_precision
        self._revision = revision

        self._model: SparseEncoder | None = None
        self._device: str | None = None
        self._sparse_dim: int | None = None

    def load(self, device: str) -> None:
        """Load the model onto the specified device.

        Args:
            device: Device string (e.g., "cuda:0", "cpu").
        """
        if (
            self._trust_remote_code
            and not Path(self._model_name_or_path).exists()
            and not is_immutable_revision(self._revision)
        ):
            raise ValueError(
                "Remote-code model loads require an immutable 40-character revision; local model paths are exempt"
            )

        self._device = device
        load_kwargs: dict[str, Any] = {
            "device": device,
            "trust_remote_code": self._trust_remote_code,
            "model_kwargs": {"torch_dtype": self._compute_precision},
        }
        # Omit revision unless pinned so the library default (main) is unchanged.
        if self._revision is not None:
            load_kwargs["revision"] = self._revision
        if self._trust_remote_code:
            self._register_remote_code_dir()
        self._model = SparseEncoder(self._model_name_or_path, **load_kwargs)

        if self._max_seq_length is not None:
            self._model.max_seq_length = self._max_seq_length

        # Sparse dim is vocabulary size
        self._sparse_dim = self._model.get_embedding_dimension()

    def _register_remote_code_dir(self) -> None:
        """Put the checkpoint's snapshot directory on ``sys.path``.

        sentence-transformers' ``Router`` resolves its sub-module classes with a
        bare ``importlib.import_module(<module>)`` (``router.load`` ->
        ``util.import_from_string``) that has no ``trust_remote_code`` snapshot
        fallback, unlike the top-level module loader. So a custom module such as
        ``modeling_st_vsplade`` only resolves when its directory is already
        importable; serving environments that don't pre-import it otherwise fail
        with ``ModuleNotFoundError`` (#2209). Registering the snapshot dir here
        makes the load deterministic across serve, smoke, and release images.

        The directory is prepended so this checkpoint's copy of the custom module
        wins, and any same-named module already cached from a *different*
        snapshot is evicted first — otherwise a sibling checkpoint that ships a
        module of the same name (e.g. two visual-SPLADE variants co-served in one
        process) would silently reuse the first one's code. Already-loaded models
        keep working: sentence-transformers holds the resolved class objects, so
        eviction only affects subsequent imports.
        """
        local = Path(self._model_name_or_path)
        if local.exists():
            base = local if local.is_dir() else local.parent
            snapshot_root = base.resolve()
        else:
            downloaded = snapshot_download(self._model_name_or_path, revision=self._revision)
            snapshot_root = Path(downloaded).resolve()

        for module_file in snapshot_root.glob("*.py"):
            cached = sys.modules.get(module_file.stem)
            cached_file = getattr(cached, "__file__", None)
            if cached_file and Path(cached_file).resolve() != module_file.resolve():
                del sys.modules[module_file.stem]

        snapshot_dir = str(snapshot_root)
        if snapshot_dir in sys.path:
            sys.path.remove(snapshot_dir)
        sys.path.insert(0, snapshot_dir)

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

        Args:
            items: List of items to encode. Each item carries either text or
                an image (first image is used, matching the pipeline's
                one-image-per-item preprocessing contract).
            output_types: Should contain "sparse".
            instruction: Optional instruction/prompt for the encoder.
            is_query: Whether items are queries (uses encode_query vs encode_document).
            prepared_items: Not used by this adapter (images are decoded from
                the raw item bytes so the SparseEncoder's own processor runs).

        Returns:
            EncodeOutput with sparse embeddings.

        Raises:
            RuntimeError: If model not loaded.
            ValueError: If output_types contains unsupported types or an item
                has neither text nor an image.
        """
        self._check_loaded()
        if self._model is None:
            raise RuntimeError(ERR_NOT_LOADED)

        validate_output_types(output_types, {"sparse"}, type(self).__name__)

        inputs = [self._extract_input(item) for item in items]

        with torch.inference_mode():
            # SparseEncoder has separate methods for query vs document
            # Use sparse COO tensor output for efficiency
            if is_query:
                embeddings = self._model.encode_query(
                    inputs,
                    prompt=instruction,
                    convert_to_tensor=True,
                    convert_to_sparse_tensor=True,
                )
            else:
                embeddings = self._model.encode_document(
                    inputs,
                    prompt=instruction,
                    convert_to_tensor=True,
                    convert_to_sparse_tensor=True,
                )

        # Convert sparse COO tensor to our format (indices + values per item)
        # embeddings is a sparse COO tensor with shape [batch, vocab_size]
        # indices[0] = row indices, indices[1] = column indices (token IDs)
        embeddings = cast("torch.Tensor", embeddings)
        sparse_indices = embeddings._indices()
        sparse_values = embeddings._values()

        sparse_list = []
        for i in range(len(items)):
            # Get entries for this item (where row index == i)
            item_mask = sparse_indices[0] == i
            token_ids = sparse_indices[1][item_mask].cpu().numpy().astype(np.int32)
            weights = sparse_values[item_mask].cpu().numpy().astype(np.float32)
            sparse_list.append(SparseVector(indices=token_ids, values=weights))

        return EncodeOutput(
            sparse=sparse_list,
            batch_size=len(items),
            is_query=is_query,
        )

    def _extract_input(self, item: Item) -> Any:
        """Return the SparseEncoder input for an item: PIL image or text."""
        if item.images:
            img_bytes = media_bytes(item.images[0], kind="image")
            pil_img = Image.open(io.BytesIO(img_bytes))
            if pil_img.mode != "RGB":
                pil_img = pil_img.convert("RGB")
            return pil_img
        if item.text is not None:
            return item.text
        raise ValueError(ERR_REQUIRES_TEXT_OR_IMAGE)
