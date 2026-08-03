"""ColSmol adapter for small, permissive visual document retrieval.

ColSmol (``vidore/colSmol-256M``, ``vidore/colSmol-500M``) is a ColPali-style
visual late-interaction retriever built on **SmolVLM** (Idefics3 architecture).
It encodes document page images into 128-dim per-patch multi-vectors and text
queries into per-token multi-vectors for MaxSim late-interaction scoring — the
small/fast permissive counterpart to the 3-4B ColQwen picks.

License (both permissive → hosted-serving-allowed):
- ColSmol adapter weights (``vidore/colSmol-*``): **MIT**
- SmolVLM backbone (``HuggingFaceTB/SmolVLM-*-Instruct``): **Apache-2.0**

The published ``vidore/colSmol-*`` repos are PEFT LoRA adapters over a
``vidore/ColSmolVLM-Instruct-*-base`` checkpoint (full ColIdefics3 weights,
including the 128-dim projection ``linear``). We resolve the base from the LoRA's
``adapter_config.json``, load it, apply the LoRA, and ``merge_and_unload`` so
inference is a plain forward. The ColIdefics3 wrapper (``Idefics3Model`` + a
``linear`` projection to 128-dim) is inlined here to avoid a colpali-engine
dependency, which conflicts with our torch>=2.9 requirement (same rationale as
the colqwen2 adapter).

Unlike the Qwen/GLM vision towers, Idefics3's SigLIP patch-embed conv runs over
full images (not pre-patched input), so the ``rebind_vision_patch_embed`` fast
path does not apply here (matching the colpali adapter).

Reference: https://github.com/illuin-tech/colpali
See: https://huggingface.co/vidore/colSmol-256M
"""

from __future__ import annotations

import io
import logging
import threading
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar

import numpy as np
import torch
from torch.nn import functional as F

from sie_server.adapters._base_adapter import BaseAdapter
from sie_server.adapters._multivector import maxsim_scores
from sie_server.adapters._spec import AdapterSpec
from sie_server.adapters._types import ComputePrecision
from sie_server.core.inference_output import EncodeOutput
from sie_server.core.postprocessor import MuveraConfig, MuveraPostprocessor
from sie_server.types.inputs import media_bytes

if TYPE_CHECKING:
    from PIL import Image as PILImage

    from sie_server.types.inputs import Item

logger = logging.getLogger(__name__)

_ERR_NO_INPUT = "ColSmolAdapter requires either text or images input"

# ColIdefics3Processor constants (from colpali-engine). Images are described with
# a fixed visual prompt; text queries are augmented with a repeated end-of-turn
# token so the query has enough tokens for a stable late-interaction match.
_VISUAL_PROMPT_PREFIX = "<|im_start|>User:<image>Describe the image.<end_of_utterance>\nAssistant:"
_QUERY_AUGMENTATION_TOKEN = "<end_of_utterance>"  # noqa: S105 — model special token, not a password
_QUERY_AUGMENTATION_COUNT = 10


# ---------------------------------------------------------------------------
# Inlined ColIdefics3 model class
# (from colpali-engine, avoids dependency conflict with torch>=2.9)
# ---------------------------------------------------------------------------


def _make_colidefics3_cls() -> type:
    """Lazily create the ColIdefics3 model class.

    Defers the transformers import so the module can be imported without
    transformers installed (for config-only operations).
    """
    from torch import nn
    from transformers import Idefics3Model, Idefics3PreTrainedModel

    class ColIdefics3(Idefics3PreTrainedModel):
        """SmolVLM (Idefics3) with a ColBERT-style multi-vector projection.

        Projects the base model's last hidden state to 128-dim, L2-normalizes,
        and masks padding via ``attention_mask``. Weight layout matches the
        ``vidore/ColSmolVLM-Instruct-*-base`` checkpoints: ``model.*`` for the
        Idefics3 tower and ``linear.*`` for the projection.

        Inlined from colpali-engine to avoid a torch version conflict.
        Reference: https://github.com/illuin-tech/colpali
        """

        main_input_name: ClassVar[str] = "doc_input_ids"

        def __init__(self, config: Any) -> None:
            super().__init__(config=config)
            self.model = Idefics3Model(config)
            self.dim = 128
            self.linear = nn.Linear(config.text_config.hidden_size, self.dim)
            self.post_init()

        def forward(self, *args: Any, **kwargs: Any) -> torch.Tensor:
            kwargs.pop("output_hidden_states", None)
            kwargs.pop("return_dict", None)
            outputs = self.model(*args, **kwargs)
            last_hidden_states = outputs.last_hidden_state  # (batch, seq, hidden_size)
            proj = self.linear(last_hidden_states)  # (batch, seq, 128)
            proj = proj / proj.norm(dim=-1, keepdim=True)  # L2 normalize
            proj = proj * kwargs["attention_mask"].unsqueeze(-1)  # mask padding
            return proj

    return ColIdefics3


class ColSmolAdapter(BaseAdapter):
    """Adapter for ColSmol visual document retrieval models (SmolVLM/Idefics3).

    Encodes document page images into 128-dim per-patch multi-vectors and text
    queries into per-token multi-vectors for MaxSim late-interaction retrieval.
    """

    spec = AdapterSpec(
        inputs=("text", "image"),
        outputs=("multivector", "score"),
        multivector_dim=128,
        unload_fields=("_model", "_processor"),
        default_preprocessor="image",
    )

    def __init__(
        self,
        model_name_or_path: str | Path,
        *,
        normalize: bool = True,
        compute_precision: ComputePrecision = "bfloat16",
        trust_remote_code: bool = False,
        revision: str | None = None,
        max_seq_length: int | None = None,
        muvera_config: dict[str, Any] | None = None,
        token_dim: int = 128,
    ) -> None:
        """Initialize the adapter.

        Args:
            model_name_or_path: HuggingFace model ID (a ``vidore/colSmol-*`` LoRA
                repo) or local path.
            normalize: Kept for interface parity — the model wrapper already
                L2-normalizes; this is a no-op safety belt for downstream parity.
            compute_precision: Compute precision for inference (float32 on CPU).
            trust_remote_code: Unused — ColSmol loads via native transformers
                Idefics3 classes (no remote code).
            revision: Optional HuggingFace revision/branch/commit SHA to pin when
                loading model artifacts. Forwarded to ``from_pretrained(..., revision=...)``.
            max_seq_length: Ignored — ColSmol uses dynamic sequence length.
            muvera_config: Optional MUVERA configuration for converting
                multi-vector outputs to dense representations.
            token_dim: Per-token embedding dimension (128 for ColSmol).
        """
        self._model_name_or_path = str(model_name_or_path)
        self._normalize = normalize
        self._compute_precision = compute_precision
        self._trust_remote_code = trust_remote_code
        self._revision = revision

        self._model: Any = None
        self._processor: Any = None
        # HF fast tokenizers are NOT thread-safe: applying per-call padding/truncation
        # reconfigures the underlying Rust tokenizer (a mutable borrow). The direct
        # adapter path (encode_pipeline.py: asyncio.to_thread, no per-model lock) lets
        # concurrent requests race with RuntimeError: Already borrowed (#2098). Serialise
        # the processor call — microseconds vs the GPU forward. Matches CLIP/SigLIP.
        self._tokenizer_lock = threading.Lock()
        self._device: str | None = None
        self._muvera_config = muvera_config
        self._multivector_dim: int = token_dim

    def load(self, device: str) -> None:
        """Load base + LoRA, merge, and place the model on ``device``."""
        from peft import PeftConfig, PeftModel
        from transformers import Idefics3Config, Idefics3Processor

        self._device = device

        dtype = self._resolve_dtype()
        attn_impl = self._resolve_attn_implementation(device)

        # The colSmol-* repos are LoRA adapters; discover the full base checkpoint
        # (which carries the ColIdefics3 projection ``linear``) from adapter_config.
        peft_cfg = PeftConfig.from_pretrained(self._model_name_or_path, revision=self._revision)
        base_id = peft_cfg.base_model_name_or_path
        if not base_id:
            msg = f"ColSmol repo {self._model_name_or_path!r} has no base_model_name_or_path in adapter_config"
            raise ValueError(msg)

        logger.info(
            "Loading ColSmol model %s (base=%s) on device=%s with dtype=%s, attn=%s",
            self._model_name_or_path,
            base_id,
            device,
            dtype,
            attn_impl,
        )

        # The pin identifies a commit in the LoRA repo (``self._model_name_or_path``);
        # guard it so the str-typed ``revision`` param keeps its "main" default when
        # unset. It applies ONLY to loads that read the LoRA repo (the processor here
        # and the PEFT config/adapter below). ``base_id`` is a *separate* base
        # checkpoint repo, so its loads must NOT receive this SHA — the commit does
        # not exist there and would raise RevisionNotFoundError (503 on first request).
        rev_kwargs: dict[str, Any] = {"revision": self._revision} if self._revision is not None else {}

        self._processor = Idefics3Processor.from_pretrained(self._model_name_or_path, **rev_kwargs)
        self._processor.tokenizer.padding_side = "left"  # ty: ignore[unresolved-attribute]

        # base_id is a different repo → load at its own default revision (no pin).
        config = Idefics3Config.from_pretrained(base_id)
        col_idefics3_cls = _make_colidefics3_cls()

        load_kwargs: dict[str, Any] = {
            "config": config,
            "dtype": dtype,
            "device_map": device,
        }
        if attn_impl is not None:
            load_kwargs["attn_implementation"] = attn_impl

        base_model = col_idefics3_cls.from_pretrained(base_id, **load_kwargs)  # ty: ignore[unresolved-attribute]
        merged = PeftModel.from_pretrained(
            base_model, self._model_name_or_path, revision=self._revision
        ).merge_and_unload()
        self._model = merged.eval()

        self._multivector_dim = getattr(self._model, "dim", 128)

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

        # One text prompt per image; nested image lists pair image i with prompt i.
        with self._tokenizer_lock:
            batch = self._processor(
                text=[_VISUAL_PROMPT_PREFIX] * len(images),
                images=[[img] for img in images],
                padding="longest",
                return_tensors="pt",
            )
        batch = {k: v.to(self._device) for k, v in batch.items() if hasattr(v, "to")}

        with torch.inference_mode():
            embeddings = self._model(**batch)  # (batch, seq, 128)
            if self._normalize:
                embeddings = F.normalize(embeddings, p=2, dim=-1)

        results = [embeddings[i].float().cpu().numpy() for i in range(embeddings.shape[0])]

        # Free GPU memory between batches to prevent OOM on subsequent calls
        # (L4 22GB GPUs are tight for VLM models).
        del embeddings, batch
        if self._device and self._device.startswith("cuda"):
            torch.cuda.empty_cache()

        return results

    # ------------------------------------------------------------------
    # Text encoding
    # ------------------------------------------------------------------

    def _encode_text(self, text: str) -> np.ndarray:
        """Encode a single text query with the query-augmentation suffix."""
        assert self._model is not None
        assert self._processor is not None

        augmented = text + _QUERY_AUGMENTATION_TOKEN * _QUERY_AUGMENTATION_COUNT
        with self._tokenizer_lock:
            batch = self._processor(
                text=[augmented],
                return_tensors="pt",
                padding="longest",
            )
        batch = {k: v.to(self._device) for k, v in batch.items() if hasattr(v, "to")}

        with torch.inference_mode():
            embeddings = self._model(**batch)  # (1, seq, 128)
            if self._normalize:
                embeddings = F.normalize(embeddings, p=2, dim=-1)

        result = embeddings[0].float().cpu().numpy()

        del embeddings, batch
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
        """Score documents against a text query using MaxSim.

        ``instruction`` and ``options`` are accepted for interface parity with
        ``BaseAdapter.score_pairs()`` but are not used: ColSmol (a ColPali-family
        model) distinguishes queries from documents via ``is_query``, not an
        instruction prefix, so a caller-supplied instruction has no effect here.
        """
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
            msg = f"Unsupported output types: {unsupported}. ColSmolAdapter only supports 'multivector'."
            raise ValueError(msg)

    def get_postprocessors(self) -> dict[str, Any]:
        """Return the configured MUVERA multivector-to-dense postprocessor."""
        config = MuveraConfig(**self._muvera_config) if self._muvera_config else MuveraConfig()
        return {"muvera": MuveraPostprocessor(token_dim=self._multivector_dim, config=config)}

    def get_preprocessor(self) -> Any | None:
        # ColSmol uses the Idefics3Processor, which requires the visual prompt
        # text alongside images; the generic ImagePreprocessor does not match
        # that call pattern (same as colqwen2/colqwen3).
        return None
