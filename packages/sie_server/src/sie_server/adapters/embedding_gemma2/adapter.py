"""Text embeddings for google/embeddinggemma-2 on the native transformers class.

EmbeddingGemma 2 is an omni embedding model (text, image, video, audio) whose
checkpoint uses ``EmbeddingGemma2Model`` with no custom code. That class first
shipped in transformers 5.19.0. Sentence Transformers 6.1 applies the official
prompt, pooling, and normalization recipe, but the transformers5 bundle pins
sentence-transformers 5.7.0, so this adapter reproduces the text profile itself:

* task prompts from ``config_sentence_transformers.json``, prepended and kept
  inside the pool (``1_Pooling`` sets ``include_prompt: true``);
* mask-aware mean pooling of ``last_hidden_state``;
* L2 normalization in float32, after any Matryoshka slice.

The text-only load sets ``vision_config`` and ``audio_config`` to ``None``,
which the model card documents as the ~270M deployment of the 744M checkpoint.
Image, audio, and video are refused: SIE's media plumbing does not produce the
processor inputs those towers consume (see :func:`_reject_non_text`).
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch
from packaging.version import Version
from torch.nn import functional as F

from sie_server.adapters._base_adapter import BaseAdapter
from sie_server.adapters._spec import AdapterSpec
from sie_server.adapters._transformers_compat import install_transformers_security_fixes
from sie_server.adapters._types import ERR_NOT_LOADED, ERR_REQUIRES_TEXT, ComputePrecision
from sie_server.core.inference_output import EncodeOutput
from sie_server.types.inputs import Item

logger = logging.getLogger(__name__)

# First transformers release that defines EmbeddingGemma2Model / embedding_gemma2.
# 5.18.0 and 5.18.1 do not; 5.19.0 (2026-10-06) does.
MIN_TRANSFORMERS = "5.19.0"

NATIVE_DIM = 768
# Card and config_sentence_transformers.json. 768 is the untruncated head.
MATRYOSHKA_DIMS = (128, 256, 512, 768)
# Shared context window from the model card. The tokenizer's model_max_length
# is an unbounded sentinel, so the catalog cap is what truncation uses.
CONTEXT_LENGTH = 8192

# Trailing space is part of the published prompt. Prompts are included in the
# mean pool; they are not a prefix that pooling strips.
QUERY_PROMPT = "task: search result | query: "
DOCUMENT_PROMPT = "title: none | text: "
QUERY_TEMPLATE = QUERY_PROMPT + "{text}"
DOCUMENT_TEMPLATE = DOCUMENT_PROMPT + "{text}"

_MODALITY_GAP = (
    "Image, audio, and video are not wired. SIE decodes images to PIL, video to "
    "sampled frames, and audio to speech PCM. EmbeddingGemma2Processor instead "
    "owns image soft-tokens (max_soft_tokens), video frames as video tokens "
    "(fps, timestamps, pixel_values_videos), and Gemma4 audio features. Those "
    "are different inputs, and this profile does not load the vision or audio towers."
)


class EmbeddingGemma2Adapter(BaseAdapter):
    """Dense text encoder for ``google/embeddinggemma-2``.

    Queries use the retrieval query prompt and documents use the document
    prompt. Both are included in mean pooling. Callers can pass
    ``options.truncate_dim`` of 128, 256, 512, or 768; the slice is
    L2-normalized again when normalization is on. Float16 is not used: the
    card reports NaN activations in that dtype.
    """

    spec = AdapterSpec(
        inputs=("text",),
        outputs=("dense",),
        dense_dim=NATIVE_DIM,
        unload_fields=("_model", "_tokenizer", "_dense_dim"),
    )

    def __init__(
        self,
        model_name_or_path: str | Path,
        *,
        normalize: bool = True,
        max_seq_length: int | None = None,
        compute_precision: ComputePrecision | None = None,
        trust_remote_code: bool = False,
        revision: str | None = None,
        query_template: str | None = QUERY_TEMPLATE,
        doc_template: str | None = DOCUMENT_TEMPLATE,
        pooling: str = "mean",
        dense_dim: int | None = None,
        **kwargs: Any,
    ) -> None:
        if trust_remote_code:
            msg = "EmbeddingGemma2Adapter does not execute Hub code; trust_remote_code must be false"
            raise ValueError(msg)
        if pooling != "mean":
            msg = f"EmbeddingGemma 2 pools with mean, not {pooling!r}"
            raise ValueError(msg)
        _ = kwargs
        self._model_name_or_path = str(model_name_or_path)
        self._normalize = normalize
        self._max_seq_length = max_seq_length or CONTEXT_LENGTH
        self._compute_precision = compute_precision
        self._revision = revision
        self._query_template = query_template
        self._doc_template = doc_template
        self._configured_dense_dim = dense_dim

        self._model: Any = None
        self._tokenizer: Any = None
        self._device: str | None = None
        self._dense_dim: int | None = dense_dim or NATIVE_DIM

    def load(self, device: str) -> None:
        import transformers

        if Version(transformers.__version__) < Version(MIN_TRANSFORMERS):
            msg = (
                f"EmbeddingGemma2Model requires transformers>={MIN_TRANSFORMERS}, "
                f"found {transformers.__version__}. The class first shipped in transformers "
                "5.19.0. This adapter is on the transformers5 bundle; transformers 4.57.6 "
                "cannot load it."
            )
            raise RuntimeError(msg)

        from transformers import AutoConfig, AutoTokenizer, EmbeddingGemma2Model  # ty: ignore[unresolved-import]

        self._device = device
        dtype = self._resolve_dtype()
        shared_kwargs: dict[str, Any] = {
            "trust_remote_code": False,
            "token": os.environ.get("HF_TOKEN"),
        }
        if self._revision is not None:
            shared_kwargs["revision"] = self._revision

        logger.info(
            "Loading EmbeddingGemma 2 %s on device=%s dtype=%s revision=%s (text towers only)",
            self._model_name_or_path,
            device,
            dtype,
            self._revision,
        )
        install_transformers_security_fixes()
        config = AutoConfig.from_pretrained(self._model_name_or_path, **shared_kwargs)
        # Card: unused towers are skipped and their checkpoint keys ignored.
        config.vision_config = None
        config.audio_config = None

        self._tokenizer = AutoTokenizer.from_pretrained(
            self._model_name_or_path,
            padding_side="right",
            **shared_kwargs,
        )
        self._model = EmbeddingGemma2Model.from_pretrained(
            self._model_name_or_path,
            config=config,
            torch_dtype=dtype,
            **shared_kwargs,
        )
        self._model.to(device)
        self._model.eval()

        text_config = self._model.config.text_config
        observed = int(getattr(text_config, "embedding_dim", NATIVE_DIM))
        self._dense_dim = self._validate_dense_dim(observed)

    def _validate_dense_dim(self, observed_dim: int) -> int:
        if self._configured_dense_dim is not None and observed_dim != self._configured_dense_dim:
            msg = (
                "EmbeddingGemma 2 dimension mismatch: "
                f"configured dense_dim={self._configured_dense_dim}, embedding_dim={observed_dim}"
            )
            raise ValueError(msg)
        return observed_dim

    def _resolve_dtype(self) -> torch.dtype:
        """bfloat16 on CUDA, float32 elsewhere. Never float16."""
        precision = self._compute_precision or "float32"
        cuda = bool(self._device and str(self._device).startswith("cuda"))
        if precision == "float16":
            logger.warning(
                "EmbeddingGemma 2 overflows float16. Using %s instead.",
                "bfloat16" if cuda else "float32",
            )
            precision = "bfloat16" if cuda else "float32"
        if not cuda:
            return torch.float32
        if precision == "bfloat16":
            return torch.bfloat16
        return torch.float32

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
        _ = instruction, prepared_items
        self._check_loaded()
        if self._model is None or self._tokenizer is None:
            raise RuntimeError(ERR_NOT_LOADED)

        unsupported = set(output_types) - {"dense"}
        if unsupported:
            msg = f"Unsupported output types: {unsupported}. This model only supports 'dense'."
            raise ValueError(msg)

        opts = options or {}
        pooling = opts.get("pooling", "mean")
        if pooling != "mean":
            msg = f"EmbeddingGemma 2 pools with mean, not {pooling!r}"
            raise ValueError(msg)
        normalize = bool(opts.get("normalize", self._normalize))
        truncate_dim = _resolve_truncate_dim(opts.get("truncate_dim", None), self._dense_dim or NATIVE_DIM)
        query_template = _option_template(opts, "query_template", self._query_template)
        doc_template = _option_template(opts, "doc_template", self._doc_template)
        template = query_template if is_query else doc_template

        for item in items:
            _reject_non_text(item)

        texts = [_format_text(template, _require_text(item)) for item in items]
        if not texts:
            width = truncate_dim or self._dense_dim or NATIVE_DIM
            return EncodeOutput(
                dense=np.zeros((0, width), dtype=np.float32),
                batch_size=0,
                is_query=is_query,
                dense_dim=width,
            )

        with self._tokenizer_guard(), torch.inference_mode():
            encoded = self._tokenizer(
                texts,
                max_length=self._max_seq_length,
                padding=True,
                truncation=True,
                return_tensors="pt",
            )
            input_ids = encoded["input_ids"].to(self._device)
            attention_mask = encoded["attention_mask"].to(self._device)
            outputs = self._model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_dict=True,
            )
            pooled = _mean_pool(outputs.last_hidden_state, attention_mask)
            if truncate_dim is not None:
                pooled = pooled[..., :truncate_dim]
            if normalize:
                pooled = F.normalize(pooled.float(), p=2, dim=-1)
            else:
                pooled = pooled.float()
            token_counts = [int(count) for count in attention_mask.sum(dim=1).tolist()]

        embeddings = pooled.cpu().numpy()
        width = int(embeddings.shape[1])
        return EncodeOutput(
            dense=embeddings,
            batch_size=len(items),
            is_query=is_query,
            dense_dim=width,
            extra={"input_token_counts": token_counts},
        )


def _require_text(item: Item) -> str:
    if item.text is None:
        raise ValueError(ERR_REQUIRES_TEXT.format(adapter_name="EmbeddingGemma2Adapter"))
    return item.text


def _reject_non_text(item: Item) -> None:
    present: list[str] = []
    if item.images:
        present.append("image")
    if item.audio is not None:
        present.append("audio")
    if item.video is not None:
        present.append("video")
    if item.document is not None:
        present.append("document")
    if present:
        msg = f"EmbeddingGemma2Adapter encodes text only in this release; got {', '.join(present)}. {_MODALITY_GAP}"
        raise ValueError(msg)


def _format_text(template: str | None, text: str) -> str:
    if not template:
        return text
    if "{text}" not in template:
        msg = f"EmbeddingGemma 2 prompt template must contain '{{text}}', got {template!r}"
        raise ValueError(msg)
    return template.format(text=text)


def _option_template(options: dict[str, Any], key: str, default: str | None) -> str | None:
    if key not in options:
        return default
    value = options[key]
    if value is None:
        return None
    if not isinstance(value, str):
        msg = f"options.{key} must be a string or null, got {type(value).__name__}"
        raise ValueError(msg)
    return value


def _resolve_truncate_dim(value: object, native_dim: int) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        msg = f"truncate_dim must be an int or null, got {value!r}"
        raise ValueError(msg)
    if value == native_dim:
        return None
    if value not in MATRYOSHKA_DIMS or value > native_dim:
        msg = f"truncate_dim must be one of {MATRYOSHKA_DIMS}, got {value}"
        raise ValueError(msg)
    return value


def _mean_pool(last_hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """Mean of non-padding tokens, in float32, matching the card's recipe."""
    mask = attention_mask.to(dtype=torch.float32).unsqueeze(-1)
    summed = (last_hidden.float() * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1e-9)
    return summed / counts
