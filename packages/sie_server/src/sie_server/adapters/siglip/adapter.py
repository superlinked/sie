"""SigLIP model adapter for image-text embedding.

This adapter provides support for SigLIP (Sigmoid Loss for Language Image Pre-training)
models that produce aligned embeddings for both images and text in a shared vector space.

Uses transformers SiglipModel with SiglipProcessor. SigLIP differs from CLIP in
using sigmoid loss instead of softmax and not having a separate projection_dim -
it uses hidden_size directly.

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
import json
import logging
import os
import re
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar, Literal

import numpy as np
import torch
from PIL import Image
from torch.nn import functional

from sie_server.adapters._base_adapter import BaseAdapter
from sie_server.adapters._spec import AdapterSpec
from sie_server.adapters._types import ComputePrecision
from sie_server.core.inference_output import EncodeOutput
from sie_server.core.preprocessor.image import ImagePreprocessor, OpenCLIPImagePreprocessor
from sie_server.types.inputs import media_bytes

if TYPE_CHECKING:
    from sie_server.types.inputs import Item

logger = logging.getLogger(__name__)


def _open_clip_token_counts(input_ids: Any) -> list[int] | None:
    """Per-row non-pad token counts for an ``open_clip`` tokenizer batch (§7.3).

    open_clip pads every text to the fixed context length with id ``0``, so the
    count of non-zero ids per row is the real billable length (content plus the
    sot/eot markers) the text tower encoded. Best-effort: any tensor quirk
    yields ``None`` so the meter falls back to its reserve estimate rather than
    mis-billing — metering must never fail inference.
    """
    try:
        return [int((row != 0).sum().item()) for row in input_ids]
    except Exception:  # noqa: BLE001 — metering must never fail inference
        return None


def _attention_mask_token_counts(
    input_ids: Any,
    attention_mask: Any,
    *,
    expected_len: int,
) -> list[int] | None:
    """Count the exact tokens selected by a shape-valid binary attention mask.

    SigLIP tokenizers commonly use the same id for ``pad_token`` and
    ``eos_token``. Counting ids unequal to the pad id would therefore omit the
    real EOS token from every non-truncated input. The processor's attention
    mask is the unambiguous record of which positions the text tower encoded.

    Best-effort: malformed/custom processor output returns ``None`` so the
    caller can use the existing padding-free tokenizer recount. Metering must
    never fail inference or accept a malformed count as settlement evidence.
    """
    try:
        input_shape = tuple(input_ids.shape)
        mask_shape = tuple(attention_mask.shape)
        if len(input_shape) != 2 or mask_shape != input_shape or input_shape[0] != expected_len:
            return None
        if not bool(torch.logical_or(attention_mask == 0, attention_mask == 1).all().item()):
            return None
        return [int(row.sum().item()) for row in attention_mask]
    except Exception:  # noqa: BLE001 — metering must never fail inference
        return None


# Cross-item preprocessing is CPU-bound (PIL decode + resize). Fan the flat
# image set across a small bounded thread pool so decode/resize overlaps.
_MAX_PREPROCESS_THREADS = 8
_DEFAULT_MAX_SEQ_LENGTH = 64
_HF_TOKENIZER_MAX_LENGTH_SENTINEL = int(1e29)
_HF_COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
_OPEN_CLIP_CONFIG_LOCK = threading.Lock()


def _feature_tensor(out: Any) -> torch.Tensor:
    """Return the pooled feature tensor from a ``get_*_features`` result.

    transformers <= 4.57.x (the Modal lane pin) returns the ``[B, dim]`` feature
    tensor directly; transformers 5.x wraps it in a ``BaseModelOutputWithPooling``
    whose ``pooler_output`` is that *same* tensor. Unwrap so the transformers
    backend is robust across the pin instead of raising ``AttributeError`` in
    ``functional.normalize``. (open_clip's ``encode_*`` always returns a tensor.)
    """
    if isinstance(out, torch.Tensor):
        return out
    pooled = getattr(out, "pooler_output", None)
    if isinstance(pooled, torch.Tensor):
        return pooled
    msg = f"unexpected get_*_features return type: {type(out).__name__}"
    raise TypeError(msg)


# Error messages
_ERR_NO_INPUT = "SiglipAdapter requires either text or images input"
_ERR_OPEN_CLIP_ID = (
    "SiglipAdapter(backend='open_clip') requires open_clip_model_id (e.g. 'hf-hub:Marqo/marqo-ecommerce-embeddings-B')"
)
_ERR_OPEN_CLIP_DIM = (
    "SiglipAdapter(backend='open_clip') requires an explicit dense_dim "
    "(open_clip does not surface embedding dim through a single config attribute)"
)
_ERR_OPEN_CLIP_REVISION = (
    "SiglipAdapter open_clip Hub checkpoints require revision to be an immutable 40-character lowercase commit SHA"
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
        revision: str | None = None,
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
            max_seq_length: Fixed text-tower token length. Catalog entries pass
                the model contract explicitly; direct construction defaults to
                SigLIP's 64-token context.
            revision: Optional HuggingFace revision/branch/commit SHA to pin
                when loading the processor and model. The ``transformers``
                backend forwards it to ``from_pretrained``. The ``open_clip``
                Hub backend requires a full commit SHA and materializes that
                exact config, tokenizer, preprocess contract, and checkpoint
                before handing local files to OpenCLIP.
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
        if max_seq_length is not None and max_seq_length <= 0:
            msg = "max_seq_length must be positive"
            raise ValueError(msg)
        self._max_seq_length = _DEFAULT_MAX_SEQ_LENGTH if max_seq_length is None else max_seq_length
        self._revision = revision
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
            if self._open_clip_model_id.startswith("hf-hub:") and (
                self._revision is None or _HF_COMMIT_RE.fullmatch(self._revision) is None
            ):
                raise ValueError(_ERR_OPEN_CLIP_REVISION)

        self._model: Any | None = None
        self._processor: Any | None = None
        self._open_clip_preprocess: Any | None = None
        self._open_clip_tokenizer: Any | None = None
        self._device: str | None = None
        self._dense_dim: int | None = None
        # HF fast tokenizers are NOT thread-safe: applying per-call
        # padding/truncation flags reconfigures the underlying Rust tokenizer,
        # which takes a mutable borrow. Concurrent text encodes (e.g. the
        # managed lane's ``@modal.concurrent`` inputs dispatched to a thread
        # pool) otherwise race with ``RuntimeError: Already borrowed``.
        # Serialise the tokenizer call — microseconds vs the GPU forward.
        # Guards both backends' tokenizers.
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

        shared_kwargs: dict[str, Any] = {"trust_remote_code": self._trust_remote_code}
        if self._revision is not None:
            shared_kwargs["revision"] = self._revision

        self._processor = SiglipProcessor.from_pretrained(
            self._model_name_or_path,
            use_fast=False,
            **shared_kwargs,
        )

        self._model = SiglipModel.from_pretrained(
            self._model_name_or_path,
            dtype=dtype,
            **shared_kwargs,
        )
        self._max_seq_length = self._resolve_transformers_max_seq_length()
        self._model.to(device)
        self._model.eval()

    def _resolve_transformers_max_seq_length(self) -> int:
        """Clamp the configured length to the loaded text tower's capacity."""
        assert self._processor is not None
        assert self._model is not None

        tokenizer = getattr(self._processor, "tokenizer", None)
        tokenizer_max = getattr(tokenizer, "model_max_length", None)
        text_config = getattr(self._model.config, "text_config", None)
        position_max = getattr(text_config, "max_position_embeddings", None)
        caps = [
            value
            for value in (tokenizer_max, position_max)
            if isinstance(value, int) and not isinstance(value, bool) and 0 < value < _HF_TOKENIZER_MAX_LENGTH_SENTINEL
        ]
        if not caps:
            return self._max_seq_length

        ceiling = min(caps)
        if self._max_seq_length > ceiling:
            logger.warning(
                "%s: configured max_seq_length=%d exceeds text-tower capacity %d "
                "(tokenizer.model_max_length=%s, text_config.max_position_embeddings=%s); "
                "clamping to %d.",
                type(self).__name__,
                self._max_seq_length,
                ceiling,
                tokenizer_max,
                position_max,
                ceiling,
            )
            return ceiling
        return self._max_seq_length

    def _load_open_clip(self, device: str, dtype: torch.dtype) -> None:
        """Load via open_clip native loader.

        The model is moved to ``device`` and cast to ``dtype`` after
        construction; this matches the transformers path's behavior and avoids
        the meta-tensor incompatibility that bites HF custom-code wrappers
        which call ``self.model.to(...)`` from inside ``__init__``.
        """
        import open_clip
        from huggingface_hub import snapshot_download

        assert self._open_clip_model_id is not None  # guarded in __init__
        if not self._open_clip_model_id.startswith("hf-hub:"):
            model, _train_preproc, val_preproc = open_clip.create_model_and_transforms(self._open_clip_model_id)
            tokenizer = open_clip.get_tokenizer(self._open_clip_model_id)
        else:
            assert self._revision is not None  # Hub identifiers require a commit SHA in __init__

            repo_id = self._open_clip_model_id.removeprefix("hf-hub:")
            snapshot_patterns = [
                "added_tokens.json",
                "config.json",
                "merges.txt",
                "open_clip_config.json",
                "open_clip_model.safetensors",
                "preprocessor_config.json",
                "special_tokens_map.json",
                "spiece.model",
                "tokenizer.json",
                "tokenizer_config.json",
                "vocab.json",
            ]
            snapshot_path = Path(
                snapshot_download(
                    repo_id=repo_id,
                    revision=self._revision,
                    allow_patterns=snapshot_patterns,
                )
            )
            weights_path = snapshot_path / "open_clip_model.safetensors"
            if not weights_path.is_file():
                # Older OpenCLIP Hub exports may have only the PyTorch checkpoint.
                snapshot_patterns[snapshot_patterns.index("open_clip_model.safetensors")] = (
                    "open_clip_pytorch_model.bin"
                )
                snapshot_path = Path(
                    snapshot_download(
                        repo_id=repo_id,
                        revision=self._revision,
                        allow_patterns=snapshot_patterns,
                    )
                )
                weights_path = snapshot_path / "open_clip_pytorch_model.bin"
            if not weights_path.is_file():
                raise RuntimeError(
                    f"pinned OpenCLIP snapshot {repo_id}@{self._revision} contains no supported checkpoint"
                )

            config_path = snapshot_path / "open_clip_config.json"
            try:
                config = json.loads(config_path.read_text(encoding="utf-8"))
                model_cfg = config["model_cfg"]
                preprocess_cfg = config["preprocess_cfg"]
                if not isinstance(model_cfg, dict) or not isinstance(preprocess_cfg, dict):
                    raise TypeError("OpenCLIP config sections must be objects")
                text_cfg = model_cfg["text_cfg"]
                if not isinstance(text_cfg, dict):
                    raise TypeError("OpenCLIP text_cfg must be an object")
                image_mean = tuple(float(value) for value in preprocess_cfg["mean"])
                image_std = tuple(float(value) for value in preprocess_cfg["std"])
                image_interpolation = str(preprocess_cfg["interpolation"])
                image_resize_mode = str(preprocess_cfg["resize_mode"])
            except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
                raise RuntimeError(f"invalid pinned OpenCLIP config for {repo_id}@{self._revision}") from exc

            # Keep tokenization in the same immutable model snapshot. The upstream
            # OpenCLIP config names a separate tokenizer repo, which would otherwise
            # reintroduce default-branch drift after the checkpoint itself is pinned.
            model_cfg = dict(model_cfg)
            model_cfg["text_cfg"] = {**text_cfg, "hf_tokenizer_name": str(snapshot_path)}
            config_name = f"sie-{repo_id.replace('/', '--')}-{self._revision}"
            with tempfile.TemporaryDirectory(prefix="sie-open-clip-") as tmp_dir:
                local_config_path = Path(tmp_dir) / f"{config_name}.json"
                local_config_path.write_text(json.dumps(model_cfg), encoding="utf-8")
                # OpenCLIP's public local-config registry is process-global. Keep
                # registration, construction, and tokenizer lookup atomic so two
                # concurrent lazy model loads cannot observe a half-updated scan.
                with _OPEN_CLIP_CONFIG_LOCK:
                    open_clip.add_model_config(local_config_path)
                    model, _train_preproc, val_preproc = open_clip.create_model_and_transforms(
                        config_name,
                        pretrained=str(weights_path),
                        image_mean=image_mean,
                        image_std=image_std,
                        image_interpolation=image_interpolation,
                        image_resize_mode=image_resize_mode,
                    )
                    tokenizer = open_clip.get_tokenizer(config_name)

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
                adapter reuses the preprocessed pixel tensors (which were
                computed in parallel on a CPU executor thread) instead of
                repeating processor work on the inference thread.

        Returns:
            EncodeOutput with dense embeddings.
        """
        # ``_check_loaded`` guards on ``_model``; backend-specific helpers
        # (``_open_clip_tokenizer`` / ``_open_clip_preprocess`` for open_clip,
        # ``_processor`` for transformers) are validated inside the encode
        # branches via local ``assert``s.
        self._check_loaded()

        self._validate_output_types(output_types)

        # The worker supplies ``prepared_items`` positionally aligned with its
        # fused ``items`` list. ``PreparedItem.original_index`` is only local to
        # the originating request and therefore repeats when dynamic batching
        # fuses concurrent requests; keying on it would overwrite tensors and
        # attach another request's image to an item.
        prepared_by_index: dict[int, Any] = {}
        if prepared_items:
            for fused_index, prepared in enumerate(prepared_items):
                payload = getattr(prepared, "payload", None)
                if payload is None:
                    continue
                pixel_values = getattr(payload, "pixel_values", None)
                if pixel_values is None:
                    continue
                prepared_by_index[fused_index] = pixel_values

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
            image_vecs = self._encode_image_items(image_indices, items, prepared_by_index)
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
        # Unit-meter seam (§7.3): stamp exact per-item TEXT-tower token counts
        # (both backends). Image items took the image tower (metered per image
        # via ``count_input_images``) and contribute 0 text tokens; only stamp
        # when at least one text item was encoded so a pure-image batch stays on
        # the image dimension. The encode pipeline forwards ``extra`` to the
        # result path for metering (§P3.5).
        if per_item_token_counts is not None:
            output.extra["input_token_counts"] = per_item_token_counts
        return output

    def _get_preprocess_pool(self) -> ThreadPoolExecutor:
        """Return the lazily-created bounded preprocessing thread pool."""
        pool = self._preprocess_pool
        if pool is None:
            with self._preprocess_pool_lock:
                pool = self._preprocess_pool
                if pool is None:
                    workers = min(_MAX_PREPROCESS_THREADS, os.cpu_count() or 1)
                    pool = ThreadPoolExecutor(max_workers=max(1, workers), thread_name_prefix="siglip-preproc")
                    self._preprocess_pool = pool
        return pool

    def _preprocess_one(self, img_input: Any) -> torch.Tensor:
        """Decode + preprocess a single image input into a ``[C, H, W]`` tensor.

        Stateless per call (PIL / processor / val_preproc release the GIL on
        the heavy work), so this is safe to fan across the preprocessing pool.
        """
        img_bytes = media_bytes(img_input, kind="image")
        pil_img = Image.open(io.BytesIO(img_bytes))
        # Convert to RGB if necessary
        if pil_img.mode != "RGB":
            pil_img = pil_img.convert("RGB")
        if self._backend == "open_clip":
            assert self._open_clip_preprocess is not None
            return self._open_clip_preprocess(pil_img)
        assert self._processor is not None
        processed = self._processor(images=pil_img, return_tensors="pt")
        return processed["pixel_values"][0]

    def _preprocess_image_batch(
        self,
        image_indices: list[int],
        items: list[Item],
        prepared_by_index: dict[int, Any],
    ) -> tuple[torch.Tensor, list[int]]:
        """Build a stacked ``[N, C, H, W]`` pixel tensor + per-item image counts.

        Reuses the framework's prepared ``[C, H, W]`` tensor for single-image
        items on the open_clip backend (matching the previous fast path); every
        other image is decoded + preprocessed, fanned across the pool. Per-image
        preprocessing is independent, so the stacked tensor is identical to
        preprocessing each image serially.
        """
        # Build the flat list of preprocessing jobs in stacked order. A job is
        # either a ready ``[C, H, W]`` tensor (open_clip prepared fast path) or
        # a raw image input to decode + preprocess.
        jobs: list[tuple[str, Any]] = []
        counts: list[int] = []
        for slot, item in enumerate(items):
            item_images = item.images or []
            counts.append(len(item_images))
            orig_index = image_indices[slot]
            prepared = prepared_by_index.get(orig_index)
            if prepared is not None and len(item_images) == 1:
                jobs.append(("ready", prepared))
            else:
                jobs.extend(("decode", img_input) for img_input in item_images)

        def run(job: tuple[str, Any]) -> torch.Tensor:
            kind, payload = job
            return payload if kind == "ready" else self._preprocess_one(payload)

        if len(jobs) == 1:
            tensors = [run(jobs[0])]
        else:
            pool = self._get_preprocess_pool()
            tensors = list(pool.map(run, jobs))
        return torch.stack(tensors, dim=0), counts

    def _encode_image_items(
        self,
        image_indices: list[int],
        all_items: list[Item],
        prepared_by_index: dict[int, Any],
    ) -> Any:
        """Encode image items as one stacked forward, mean-pooling per item.

        Returns a ``[len(image_indices), dim]`` float32 array, one row per item.
        """
        assert self._model is not None

        items = [all_items[i] for i in image_indices]
        pixel_values, counts = self._preprocess_image_batch(image_indices, items, prepared_by_index)

        if self._backend == "open_clip":
            # open_clip's encode_image does not auto-cast inputs; match the
            # model dtype (e.g. fp16 on CUDA) as the serial path did.
            model_dtype = next(self._model.parameters()).dtype
            pixel_values = pixel_values.to(device=self._device, dtype=model_dtype)
            with torch.inference_mode():
                image_features = self._model.encode_image(pixel_values, normalize=self._normalize)
        else:
            pixel_values = pixel_values.to(self._device)
            with torch.inference_mode():
                image_features = _feature_tensor(self._model.get_image_features(pixel_values=pixel_values))
                if self._normalize:
                    image_features = functional.normalize(image_features, p=2, dim=-1)

        # Split by per-item image counts; mean-pool multi-image items on-device
        # in the model dtype, exactly as the serial path did.
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
        tower encoded (§7.3, both backends), or ``None`` when no clean count
        could be surfaced.
        """
        assert self._model is not None

        if self._backend == "open_clip":
            assert self._open_clip_tokenizer is not None
            # Serialise tokenization: fast tokenizers are not thread-safe.
            with self._tokenizer_lock:
                input_ids = self._open_clip_tokenizer(list(texts))
                # Unit-meter seam (§7.3): open_clip pads to a fixed context
                # length with id 0, so the non-pad count per row is the real
                # billable length (content + sot/eot) the tower encoded.
                token_counts = _open_clip_token_counts(input_ids)
                input_ids = input_ids.to(self._device)
            with torch.inference_mode():
                text_features = self._model.encode_text(input_ids, normalize=self._normalize)
        else:
            assert self._processor is not None
            # Process text as one batch - use max_length padding to match MTEB
            # behavior. SigLIP text embeddings depend on sequence length, so
            # consistent padding is required (per-item results are unchanged by
            # batching since max_length pads every item to the same length).
            # The HF fast tokenizer is not thread-safe under the per-call
            # padding/truncation reconfiguration, so serialise this call.
            with self._tokenizer_lock:
                inputs = self._processor(
                    text=list(texts),
                    return_tensors="pt",
                    padding="max_length",
                    truncation=True,
                    max_length=self._max_seq_length,
                )
                # Unit-meter seam (§7.3): the exact attention mask includes the
                # real EOS token even when it shares an id with padding (the
                # stock SigLIP tokenizer does this). Older/custom processors
                # without a shape-valid binary mask retain the conservative
                # padding-free recount fallback.
                token_counts = _attention_mask_token_counts(
                    inputs.get("input_ids"),
                    inputs.get("attention_mask"),
                    expected_len=len(texts),
                )
                if token_counts is None:
                    token_counts = self._token_counts_or_none(
                        self._processor.tokenizer, list(texts), expected_len=len(texts)
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

            return OpenCLIPImagePreprocessor(self._open_clip_preprocess, self._model_name_or_path)

        if self._processor is None:
            return None

        return ImagePreprocessor(self._processor, self._model_name_or_path)

    def unload(self) -> None:
        """Shut down the preprocessing pool, then unload model weights."""
        pool = self._preprocess_pool
        self._preprocess_pool = None
        if pool is not None:
            pool.shutdown(wait=True)
        super().unload()
