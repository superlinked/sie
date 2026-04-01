r"""PyTorch-based embedding model adapter.

This adapter provides direct PyTorch/transformers inference for embedding models,
with configurable pooling strategies and attention implementations.

Supports:
- Decoder-based models (Qwen3-Embedding, Mistral-embed, NV-Embed, etc.)
- Encoder-based models (when using CLS or mean pooling)
- Configurable attention: SDPA (default) or Flash Attention 2

Performance notes (see DESIGN.md Section 6.3):
- SDPA is 20-25% faster for sequences <512 tokens
- FA2 is faster for sequences >512 tokens
- Default is SDPA since standard retrieval workloads have short texts

Example configurations:

    # Qwen3-Embedding (decoder, last-token pooling, instruction template)
    PyTorchEmbeddingAdapter(
        model_name_or_path="Qwen/Qwen3-Embedding-0.6B",
        pooling="last_token",
        query_template="Instruct: {instruction}\nQuery: {text}",
    )

    # Mistral-embed (decoder, last-token pooling, no template)
    PyTorchEmbeddingAdapter(
        model_name_or_path="intfloat/e5-mistral-7b-instruct",
        pooling="last_token",
    )

    # Generic encoder model (CLS pooling)
    PyTorchEmbeddingAdapter(
        model_name_or_path="BAAI/bge-base-en-v1.5",
        pooling="cls",
    )
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import torch
from torch.nn import functional
from transformers import AutoModel, AutoTokenizer

from sie_server.adapters.base import ModelAdapter, ModelCapabilities, ModelDims
from sie_server.adapters.peft_lora_mixin import PEFTLoRAMixin
from sie_server.core.inference_output import EncodeOutput
from sie_server.core.preprocessor import CharCountPreprocessor
from sie_server.types.inputs import Item

if TYPE_CHECKING:
    from transformers import PreTrainedModel, PreTrainedTokenizerFast

logger = logging.getLogger(__name__)

# Type aliases
ComputePrecision = Literal["float16", "bfloat16", "float32"]
AttnImplementation = Literal["sdpa", "flash_attention_2", "eager"]
PoolingStrategy = Literal["last_token", "cls", "mean"]

# Error messages
_ERR_NOT_LOADED = "Model not loaded. Call load() first."
_ERR_REQUIRES_TEXT = "PyTorchEmbeddingAdapter requires text input"


class PyTorchEmbeddingAdapter(PEFTLoRAMixin, ModelAdapter):
    """Generic PyTorch adapter for embedding models.

    This adapter uses direct PyTorch/transformers inference with configurable
    pooling strategies and attention implementations. Works with both encoder
    and decoder architectures.

    LoRA Support:
        This adapter inherits from PEFTLoRAMixin, providing PEFT-based LoRA
        support. LoRAs are loaded via load_lora() and switched via set_active_lora().
        See peft_lora_mixin.py for implementation details.
    """

    def __init__(
        self,
        model_name_or_path: str | Path,
        *,
        pooling: PoolingStrategy = "last_token",
        normalize: bool = True,
        max_seq_length: int = 8192,
        compute_precision: ComputePrecision = "bfloat16",
        attn_implementation: AttnImplementation = "sdpa",
        trust_remote_code: bool = True,
        query_template: str | None = None,
        doc_template: str | None = None,
        default_instruction: str | None = None,
        forward_kwargs: dict[str, Any] | None = None,
        uses_legacy_transformers_cache: bool = False,
    ) -> None:
        r"""Initialize the adapter.

        Args:
            model_name_or_path: HuggingFace model ID or local path.
            pooling: Pooling strategy for embeddings:
                - "last_token": Use last non-padding token (decoder models)
                - "cls": Use [CLS] token at position 0 (encoder models)
                - "mean": Mean of all non-padding tokens
            normalize: Whether to L2-normalize embeddings.
            max_seq_length: Maximum sequence length.
            compute_precision: Compute precision (bfloat16 recommended for CUDA).
            attn_implementation: Attention implementation. Default "sdpa".
                Use "flash_attention_2" only if inputs are consistently >512 tokens.
                Use "eager" for models that don't support SDPA (e.g., Stella).
                For typical retrieval workloads, SDPA is 20-25% faster.
            trust_remote_code: Whether to trust remote code in model files.
            query_template: Template for formatting queries. Use {instruction} and
                {text} placeholders. Example: "Instruct: {instruction}\nQuery: {text}"
                If None, queries are passed as-is.
            doc_template: Template for formatting documents. Use {text} placeholder.
                If None, documents are passed as-is.
            default_instruction: Default instruction when query_template uses
                {instruction} but none is provided. Example: "Given a query,
                retrieve relevant passages"
            forward_kwargs: Extra keyword arguments to pass to the model's forward
                method. Useful for models with custom parameters like GritLM's
                `is_causal=False` for bidirectional attention in embedding mode.
            uses_legacy_transformers_cache: If True, disable the KV cache after
                loading by setting model.config.use_cache = False. Required for
                models that use the legacy transformers cache API (pre-4.54).
        """
        self._model_name_or_path = str(model_name_or_path)
        self._pooling = pooling
        self._normalize = normalize
        self._max_seq_length = max_seq_length
        self._compute_precision = compute_precision
        self._attn_implementation = attn_implementation
        self._trust_remote_code = trust_remote_code
        self._query_template = query_template
        self._doc_template = doc_template
        self._default_instruction = default_instruction
        self._forward_kwargs = forward_kwargs or {}
        self._uses_legacy_transformers_cache = uses_legacy_transformers_cache

        self._model: PreTrainedModel | None = None
        self._tokenizer: PreTrainedTokenizerFast | None = None
        self._device: str | None = None
        self._dense_dim: int | None = None

    @property
    def capabilities(self) -> ModelCapabilities:
        """Return model capabilities."""
        return ModelCapabilities(
            inputs=["text"],
            outputs=["dense"],
        )

    @property
    def dims(self) -> ModelDims:
        """Return model dimensions."""
        if self._dense_dim is None:
            raise RuntimeError(_ERR_NOT_LOADED)
        return ModelDims(dense=self._dense_dim)

    def load(self, device: str) -> None:
        """Load the model onto the specified device.

        Args:
            device: Device string (e.g., "cuda:0", "cpu").
        """
        self._device = device

        # Determine dtype and attention implementation
        dtype, attn_impl = self._resolve_dtype_and_attn(device)

        logger.info(
            "Loading %s on device=%s with dtype=%s, attn=%s, pooling=%s",
            self._model_name_or_path,
            device,
            dtype,
            attn_impl,
            self._pooling,
        )

        # Determine padding side based on pooling strategy
        padding_side = "left" if self._pooling == "last_token" else "right"

        hf_token = os.environ.get("HF_TOKEN")

        # Load tokenizer
        self._tokenizer = AutoTokenizer.from_pretrained(
            self._model_name_or_path,
            padding_side=padding_side,
            trust_remote_code=self._trust_remote_code,
            token=hf_token,
        )

        # Load model with configured attention implementation
        self._model = AutoModel.from_pretrained(
            self._model_name_or_path,
            torch_dtype=dtype,
            attn_implementation=attn_impl,
            trust_remote_code=self._trust_remote_code,
            token=hf_token,
        )
        # Disable KV cache for models using the legacy transformers cache API
        if self._uses_legacy_transformers_cache:
            self._model.config.use_cache = False
        self._model.to(device)
        self._model.eval()

        # Get embedding dimension from model config
        self._dense_dim = self._model.config.hidden_size

    def _resolve_dtype_and_attn(self, device: str) -> tuple[torch.dtype, str]:
        """Resolve dtype and attention implementation based on device and config.

        Returns:
            Tuple of (torch.dtype, attention_implementation string).
        """
        # CPU should use FP32; respect explicit eager but default to SDPA
        if not device.startswith("cuda"):
            return torch.float32, self._attn_implementation if self._attn_implementation == "eager" else "sdpa"

        # Map precision to dtype
        dtype_map = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }
        dtype = dtype_map.get(self._compute_precision, torch.bfloat16)

        # Use configured attention implementation directly
        return dtype, self._attn_implementation

    def unload(self) -> None:
        """Unload the model and free resources."""
        device = self._device  # Save before clearing

        if self._model is not None:
            del self._model
            self._model = None

        if self._tokenizer is not None:
            del self._tokenizer
            self._tokenizer = None

        self._device = None
        self._dense_dim = None

        # Release GPU memory per the memory management contract in base.py
        import gc

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
        prepared_items: Any = None,
        options: dict[str, Any] | None = None,
    ) -> EncodeOutput:
        """Run inference returning standardized batched output.

        Args:
            items: List of items to encode.
            output_types: Which outputs to return (only "dense" supported).
            instruction: Optional instruction for queries.
            is_query: Whether items are queries (True) or documents (False).
            prepared_items: Not used by this adapter.

        Returns:
            EncodeOutput with dense embeddings.
        """
        if self._model is None or self._tokenizer is None:
            raise RuntimeError(_ERR_NOT_LOADED)

        self._validate_output_types(output_types)

        # Resolve runtime options (config defaults -> profile -> request overrides)
        opts = options or {}
        query_template = opts.get("query_template", self._query_template)
        doc_template = opts.get("doc_template", self._doc_template)
        default_instruction = opts.get("default_instruction", self._default_instruction)
        normalize = opts.get("normalize", self._normalize)
        pooling = opts.get("pooling", self._pooling)

        # Validate pooling safety: last_token requires left padding (set at load time)
        if pooling == "last_token" and self._pooling != "last_token":
            msg = (
                "Cannot use 'last_token' pooling at runtime when the model was loaded "
                f"with '{self._pooling}' pooling (right padding). last_token pooling "
                "requires left padding which is configured at load time."
            )
            raise ValueError(msg)

        texts = self._format_texts(
            items,
            instruction,
            is_query=is_query,
            query_template=query_template,
            doc_template=doc_template,
            default_instruction=default_instruction,
        )

        # Tokenize
        inputs = self._tokenizer(
            texts,
            max_length=self._max_seq_length,
            padding=True,
            truncation=True,
            return_tensors="pt",
        ).to(self._device)

        # Forward pass
        with torch.inference_mode():
            outputs = self._model(**inputs, return_dict=True, **self._forward_kwargs)

            # Handle models that return sentence_embeddings dict (e.g. NV-Embed-v2)
            if isinstance(outputs, dict) and "sentence_embeddings" in outputs:
                hidden_state = outputs["sentence_embeddings"]
                attention_mask = inputs["attention_mask"]
                embeddings = self._apply_pooling(hidden_state, attention_mask, pooling=pooling)
            else:
                last_hidden_state = outputs.last_hidden_state
                attention_mask = inputs["attention_mask"]
                embeddings = self._apply_pooling(last_hidden_state, attention_mask, pooling=pooling)

            # L2 normalize if configured
            if normalize:
                embeddings = functional.normalize(embeddings, p=2, dim=-1)

        embeddings_np = embeddings.float().cpu().numpy()
        return EncodeOutput(
            dense=embeddings_np,
            batch_size=len(items),
            is_query=is_query,
            dense_dim=self._dense_dim,
        )

    def _apply_pooling(
        self,
        hidden_state: torch.Tensor,
        attention_mask: torch.Tensor,
        *,
        pooling: str | None = None,
    ) -> torch.Tensor:
        """Apply the configured pooling strategy.

        Args:
            hidden_state: Model output [batch, seq_len, hidden_dim]
            attention_mask: Attention mask [batch, seq_len]
            pooling: Override pooling strategy (falls back to self._pooling).

        Returns:
            Pooled embeddings [batch, hidden_dim]
        """
        pooling = pooling if pooling is not None else self._pooling

        if pooling == "cls":
            # CLS token is always at position 0
            return hidden_state[:, 0]

        if pooling == "mean":
            # Mean of non-padding tokens
            mask = attention_mask.unsqueeze(-1).float()
            sum_embeddings = (hidden_state * mask).sum(dim=1)
            sum_mask = mask.sum(dim=1).clamp(min=1e-9)
            return sum_embeddings / sum_mask

        # last_token pooling
        # With left padding (padding_side="left"), the last token is always
        # at the final position. Simply take hidden_state[:, -1].
        # Note: This assumes left padding which is set in load() when pooling="last_token"
        return hidden_state[:, -1]

    def _validate_output_types(self, output_types: list[str]) -> None:
        """Validate that output types are supported."""
        unsupported = set(output_types) - {"dense"}
        if unsupported:
            msg = f"Unsupported output types: {unsupported}. This model only supports 'dense'."
            raise ValueError(msg)

    def _format_texts(
        self,
        items: list[Item],
        instruction: str | None,
        *,
        is_query: bool,
        query_template: str | None = None,
        doc_template: str | None = None,
        default_instruction: str | None = None,
    ) -> list[str]:
        r"""Format texts using configured templates.

        For queries with query_template, formats using the template.
        For documents with doc_template, formats using the template.
        Otherwise returns text as-is.
        """
        query_template = query_template if query_template is not None else self._query_template
        doc_template = doc_template if doc_template is not None else self._doc_template
        default_instruction = default_instruction if default_instruction is not None else self._default_instruction
        texts = []
        for item in items:
            if item.text is None:
                raise ValueError(_ERR_REQUIRES_TEXT)

            text = item.text

            if is_query and query_template:
                # Use provided instruction or default
                instr = instruction or default_instruction or ""
                text = query_template.format(instruction=instr, text=text)
            elif not is_query and doc_template:
                text = doc_template.format(text=text)
            elif instruction:
                # Fallback: prepend instruction if provided but no template
                text = f"{instruction} {text}"

            texts.append(text)
        return texts

    def get_preprocessor(self) -> CharCountPreprocessor:
        """Return CharCountPreprocessor for cost estimation without tokenization overhead."""
        return CharCountPreprocessor(model_name=self._model_name_or_path)
