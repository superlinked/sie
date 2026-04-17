from __future__ import annotations

import importlib.util
import logging
from typing import TYPE_CHECKING, Any, ClassVar, Literal, cast

import numpy as np
import torch

from sie_server.adapters._flash_base import FlashBaseAdapter
from sie_server.adapters._spec import AdapterSpec
from sie_server.adapters._types import ERR_NOT_LOADED, ERR_REQUIRES_TEXT, ComputePrecision
from sie_server.adapters._utils import apply_rotary_pos_emb, extract_texts, validate_output_types
from sie_server.adapters.peft_lora_mixin import PEFTLoRAMixin
from sie_server.core.inference_output import EncodeOutput, SparseVector
from sie_server.types.inputs import Item

if TYPE_CHECKING:
    from pathlib import Path

    from transformers import PreTrainedTokenizerFast

logger = logging.getLogger(__name__)

_ERR_WRONG_ARCH = "GTESparseFlashAdapter requires NewForMaskedLM architecture (model.new attribute)"


class GTESparseFlashAdapter(PEFTLoRAMixin, FlashBaseAdapter):
    """GTE sparse flash adapter for NewForMaskedLM architecture.

    This adapter uses Flash Attention 2's variable-length attention for efficient
    batched inference without padding waste on GPU. Falls back to native forward
    on CPU.

    Architecture: POST-norm (LayerNorm after residual, not before sublayer).

    Produces SPLADE-style sparse lexical representations using masked language modeling.

    Supports LoRA adapters via PEFTLoRAMixin.
    """

    fallback_adapter_path: ClassVar[str | None] = None

    spec = AdapterSpec(
        inputs=("text",),
        outputs=("sparse",),
        unload_fields=(
            "_model",
            "_tokenizer",
            "_vocab_size",
            "_num_heads",
            "_head_dim",
            "_hidden_size",
            "_idf",
            "_use_flash",
            "_activation_mode",
            "_special_token_ids",
        ),
    )

    def _resolve_dtype(self) -> torch.dtype:
        """Resolve dtype; force float32 on CPU for numerical stability."""
        if self._device == "cpu":
            return torch.float32
        return super()._resolve_dtype()

    def __init__(
        self,
        model_name_or_path: str | Path,
        *,
        max_seq_length: int = 512,
        compute_precision: ComputePrecision = "float16",
        query_template: str | None = None,
        doc_template: str | None = None,
        trust_remote_code: bool = True,
        **kwargs: Any,
    ) -> None:
        """Initialize the adapter.

        Args:
            model_name_or_path: HuggingFace model ID or local path.
            max_seq_length: Maximum sequence length.
            compute_precision: Compute precision (float16/bfloat16/float32).
            query_template: Optional template for queries.
            doc_template: Optional template for documents.
            trust_remote_code: Whether to trust remote code (required for NewForMaskedLM).
            **kwargs: Additional arguments (ignored, for compatibility).
        """
        _ = kwargs
        self._model_name_or_path = str(model_name_or_path)
        self._max_seq_length = max_seq_length
        self._compute_precision = compute_precision
        self._query_template = query_template
        self._doc_template = doc_template
        self._trust_remote_code = trust_remote_code

        self._model: Any = None
        self._tokenizer: PreTrainedTokenizerFast | None = None
        self._device: str | None = None
        self._vocab_size: int | None = None
        self._num_heads: int | None = None
        self._head_dim: int | None = None
        self._hidden_size: int | None = None
        self._use_flash: bool = False
        self._idf: torch.Tensor | None = None
        self._activation_mode: Literal["v1", "v3"] = "v1"
        self._special_token_ids: list[int] = []

    def load(self, device: str) -> None:
        """Load the model onto the specified device.

        Args:
            device: Device string ("cuda", "cuda:X", or "cpu").

        Raises:
            ValueError: If model is not NewForMaskedLM architecture.
        """
        from transformers import AutoModelForMaskedLM, AutoTokenizer

        self._device = device
        self._use_flash = device.startswith("cuda") and importlib.util.find_spec("flash_attn") is not None
        dtype = self._resolve_dtype()

        attn_mode = "flash_varlen" if self._use_flash else "native"
        logger.info(
            "Loading %s on device=%s with dtype=%s, attn=%s (GTE sparse)",
            self._model_name_or_path,
            device,
            dtype,
            attn_mode,
        )

        self._tokenizer = AutoTokenizer.from_pretrained(
            self._model_name_or_path,
            trust_remote_code=self._trust_remote_code,
        )

        self._model = AutoModelForMaskedLM.from_pretrained(
            self._model_name_or_path,
            torch_dtype=dtype,
            trust_remote_code=self._trust_remote_code,
        )
        self._model.to(device)
        self._model.eval()

        # Verify this is NewForMaskedLM architecture
        if not hasattr(self._model, "new"):
            raise ValueError(_ERR_WRONG_ARCH)

        self._vocab_size = self._model.config.vocab_size
        self._sparse_dim = self._vocab_size
        self._num_heads = cast("int", self._model.config.num_attention_heads)
        self._hidden_size = cast("int", self._model.config.hidden_size)
        self._head_dim = self._hidden_size // self._num_heads

        logger.info(
            "Loaded GTE sparse: vocab_size=%d, hidden_size=%d, num_heads=%d, head_dim=%d",
            self._vocab_size,
            self._hidden_size,
            self._num_heads,
            self._head_dim,
        )
        self._idf = self._try_load_idf_vector(self._tokenizer)
        self._special_token_ids = sorted(set(self._get_special_token_ids_modelcard()))
        if "v3" in self._model_name_or_path:
            self._activation_mode = "v3"
        else:
            self._activation_mode = "v1"

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

        Uses flash_attn_varlen_func on GPU for efficient batched processing.
        Falls back to native forward on CPU.

        Args:
            items: List of items to encode.
            output_types: Which outputs to compute (only "sparse" supported).
            instruction: Optional instruction prefix.
            is_query: Whether items are queries (affects template selection).
            prepared_items: Not used by this adapter.

        Returns:
            EncodeOutput with sparse embeddings.
        """
        self._check_loaded()
        if self._tokenizer is None:
            raise RuntimeError(ERR_NOT_LOADED)

        validate_output_types(output_types, {"sparse"}, "GTESparseFlashAdapter")

        # Resolve runtime options (config defaults -> profile -> request overrides)
        opts = options or {}
        query_template = opts.get("query_template", self._query_template)
        doc_template = opts.get("doc_template", self._doc_template)

        texts = extract_texts(
            items,
            instruction,
            is_query=is_query,
            query_template=query_template,
            doc_template=doc_template,
            err_msg=ERR_REQUIRES_TEXT.format(adapter_name="GTESparseFlashAdapter"),
        )

        # Inference-free query encoding via IDF lookup (doc-* checkpoint pattern)
        if is_query and self._idf is not None:
            return self._encode_query_idf(texts, is_query)

        if self._use_flash:
            return self._encode_flash(texts, is_query)
        return self._encode_native(texts, is_query)

    def _encode_native(self, texts: list[str], is_query: bool) -> EncodeOutput:
        """Encode using native forward pass (for CPU or fallback)."""
        self._check_loaded()
        if self._tokenizer is None:
            raise RuntimeError(ERR_NOT_LOADED)
        inputs = self._tokenizer(
            texts,
            max_length=self._max_seq_length,
            truncation=True,
            padding=True,
            return_tensors="pt",
        )
        inputs = {k: v.to(self._device) for k, v in inputs.items()}

        with torch.inference_mode():
            outputs = self._model(**inputs)
            logits = outputs.logits  # [B, L, V]
            mask = inputs["attention_mask"].to(logits.device).unsqueeze(-1)  # [B, L, 1]
            values, _ = torch.max(logits.float() * mask, dim=1)  # [B, V]
            values = self._sparse_activation(values)
            special_ids = self._get_special_token_ids_modelcard()
            if special_ids:
                values[:, special_ids] = 0.0

            # Build SparseVector directly (no dict intermediary)
            sparse_list: list[SparseVector] = []
            for i in range(values.size(0)):
                nonzero_mask = values[i] > 0
                indices_i = torch.where(nonzero_mask)[0]
                vals_i = values[i, indices_i]
                sparse_list.append(
                    SparseVector(
                        indices=indices_i.cpu().numpy().astype(np.int32),
                        values=vals_i.cpu().float().numpy(),
                    )
                )

        return EncodeOutput(sparse=sparse_list, batch_size=len(texts), is_query=is_query)

    def _encode_flash(self, texts: list[str], is_query: bool) -> EncodeOutput:
        """Encode using flash attention with packed sequences."""
        from flash_attn import flash_attn_varlen_func  # ty: ignore[unresolved-import]

        self._check_loaded()
        if self._tokenizer is None:
            raise RuntimeError(ERR_NOT_LOADED)

        # Batch tokenize all texts at once (no padding for packing)
        batch_encoding = self._tokenizer(
            texts,
            max_length=self._max_seq_length,
            truncation=True,
            padding=False,
            return_attention_mask=False,
            return_token_type_ids=False,
        )

        # Build packed representation from batch encoding
        seq_lengths = [len(ids) for ids in batch_encoding["input_ids"]]
        total_tokens = sum(seq_lengths)
        max_seqlen = max(seq_lengths)

        # Pack input_ids into a single 1-D tensor
        input_ids_packed = torch.tensor(
            [tok_id for ids in batch_encoding["input_ids"] for tok_id in ids],
            dtype=torch.long,
            device=self._device,
        )

        # Build cu_seqlens using cumsum
        cu_seqlens = torch.zeros(len(texts) + 1, dtype=torch.int32, device=self._device)
        cu_seqlens[1:] = torch.tensor(seq_lengths, dtype=torch.int32, device=self._device).cumsum(0)

        with torch.inference_mode():
            # Build position IDs for RoPE
            position_ids = self._build_position_ids(cu_seqlens)

            # Compute RoPE cos/sin
            cos, sin = self._compute_rope(position_ids, max_seqlen)

            # Run embeddings
            hidden = self._run_embeddings(input_ids_packed)

            # Run transformer with flash attention (POST-norm architecture)
            hidden = self._run_transformer_flash(
                hidden, cu_seqlens, max_seqlen, total_tokens, cos, sin, flash_attn_varlen_func
            )

            # Run MLM head
            logits = self._model.lm_head(hidden)  # [total_tokens, V]

            # Compute activation weights on full tensor
            weights = self._sparse_activation(logits.float())

            # Max-pool over tokens per sequence to get sparse vectors
            sparse_list = self._aggregate_sparse(weights, cu_seqlens, seq_lengths)

        return EncodeOutput(sparse=sparse_list, batch_size=len(texts), is_query=is_query)

    def _build_position_ids(self, cu_seqlens: torch.Tensor) -> torch.Tensor:
        """Build position IDs for packed sequences (each starts from 0)."""
        total_tokens = int(cu_seqlens[-1].item())
        positions = torch.arange(total_tokens, device=self._device)
        offsets = torch.repeat_interleave(
            cu_seqlens[:-1],
            cu_seqlens[1:] - cu_seqlens[:-1],
        )
        return positions - offsets

    def _aggregate_sparse(
        self,
        weights: torch.Tensor,
        cu_seqlens: torch.Tensor,
        seq_lengths: list[int],
    ) -> list[SparseVector]:
        """Aggregate token weights to sparse vectors via max-pooling."""
        num_seqs = len(seq_lengths)
        max_weights = torch.segment_reduce(weights, "max", offsets=cu_seqlens)
        if self._special_token_ids:
            max_weights[:, self._special_token_ids] = 0.0
        dense = max_weights.cpu().float().numpy()
        results: list[SparseVector] = []
        for i in range(num_seqs):
            row = dense[i]
            mask = row > 0
            results.append(
                SparseVector(
                    indices=np.where(mask)[0].astype(np.int32),
                    values=row[mask],
                )
            )
        return results

    def _compute_rope(
        self,
        position_ids: torch.Tensor,
        max_seqlen: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute RoPE cos/sin values for packed positions."""
        rotary_emb = self._model.new.embeddings.rotary_emb
        dtype = self._resolve_dtype()

        # Ensure cache is large enough
        dummy_x = torch.zeros(1, max_seqlen, 1, device=self._device, dtype=dtype)
        _ = rotary_emb(dummy_x, seq_len=max_seqlen)

        # Index into cached values
        cos = rotary_emb.cos_cached[position_ids].to(dtype)
        sin = rotary_emb.sin_cached[position_ids].to(dtype)

        return cos, sin

    def _run_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Get token embeddings for packed input."""
        embeddings = self._model.new.embeddings
        hidden = embeddings.word_embeddings(input_ids)
        hidden = embeddings.LayerNorm(hidden)
        hidden = embeddings.dropout(hidden)
        return hidden

    def _run_transformer_flash(
        self,
        hidden: torch.Tensor,
        cu_seqlens: torch.Tensor,
        max_seqlen: int,
        total_tokens: int,
        cos: torch.Tensor,
        sin: torch.Tensor,
        flash_attn_varlen_func: Any,
    ) -> torch.Tensor:
        """Run transformer layers with flash attention.

        NewForMaskedLM uses POST-norm architecture:
        - hidden = hidden + attention(hidden)
        - hidden = attn_ln(hidden)  # POST-norm
        - hidden = hidden + mlp(hidden)
        - hidden = mlp_ln(hidden)  # POST-norm
        """
        self._check_loaded()
        if self._head_dim is None:
            raise RuntimeError(ERR_NOT_LOADED)
        softmax_scale = 1.0 / (self._head_dim**0.5)

        for layer in self._model.new.encoder.layer:
            attn = layer.attention

            # QKV projection (no pre-norm in this architecture)
            qkv = attn.qkv_proj(hidden)
            qkv = qkv.view(total_tokens, 3, self._num_heads, self._head_dim)
            query = qkv[:, 0]
            key = qkv[:, 1]
            value = qkv[:, 2]

            # Apply RoPE
            query, key = apply_rotary_pos_emb(query, key, cos, sin)

            # Flash attention
            attn_out = flash_attn_varlen_func(
                query,
                key,
                value,
                cu_seqlens_q=cu_seqlens,
                cu_seqlens_k=cu_seqlens,
                max_seqlen_q=max_seqlen,
                max_seqlen_k=max_seqlen,
                causal=False,
                softmax_scale=softmax_scale,
            )
            attn_out = attn_out.reshape(total_tokens, self._hidden_size)

            # Output projection + dropout
            attn_out = attn.o_proj(attn_out)
            attn_out = layer.hidden_dropout(attn_out)

            # Residual + POST-norm
            hidden = hidden + attn_out
            hidden = layer.attn_ln(hidden)

            # MLP (no pre-norm)
            mlp_out = layer.mlp(hidden)
            mlp_out = layer.hidden_dropout(mlp_out)

            # Residual + POST-norm
            hidden = hidden + mlp_out
            hidden = layer.mlp_ln(hidden)

        return hidden

    def _get_special_token_ids_modelcard(self) -> list[int]:
        self._check_loaded()
        if self._tokenizer is None:
            raise RuntimeError(ERR_NOT_LOADED)
        ids: list[int] = []
        for tok in self._tokenizer.special_tokens_map.values():
            if isinstance(tok, list):
                for t in tok:
                    tid = self._tokenizer.vocab.get(t)
                    if tid is not None:
                        ids.append(int(tid))
            else:
                tid = self._tokenizer.vocab.get(tok)
                if tid is not None:
                    ids.append(int(tid))
        return ids

    def _sparse_activation(self, values: torch.Tensor) -> torch.Tensor:
        """Apply SPLADE activation. v3 uses log1p(log1p(relu(.))); v1 uses log1p(relu(.))."""
        if self._activation_mode == "v3":
            return torch.log1p(torch.log1p(torch.relu_(values)))
        return torch.log1p(torch.relu_(values))

    def _try_load_idf_vector(self, tokenizer: PreTrainedTokenizerFast) -> torch.Tensor | None:
        import json
        from pathlib import Path

        p = Path(self._model_name_or_path)
        if p.exists() and p.is_dir():
            idf_path = p / "idf.json"
            if not idf_path.exists():
                logger.warning("IDF not loaded for %s: idf.json not found at %s", self._model_name_or_path, idf_path)
                return None
            with open(idf_path, encoding="utf-8") as f:
                idf = json.load(f)
        else:
            try:
                from huggingface_hub import try_to_load_from_cache

                cached_path = try_to_load_from_cache(
                    repo_id=self._model_name_or_path,
                    filename="idf.json",
                )
                # try_to_load_from_cache returns a str path, None (not cached),
                # or _CACHED_NO_EXIST sentinel (explicitly absent). Only a str
                # means the file is present locally.
                if not isinstance(cached_path, str):
                    # Not in local cache — try downloading
                    from huggingface_hub import hf_hub_download

                    cached_path = hf_hub_download(
                        repo_id=self._model_name_or_path,
                        filename="idf.json",
                    )
                if not isinstance(cached_path, str):
                    logger.warning("IDF not found for %s", self._model_name_or_path)
                    return None
                with open(cached_path, encoding="utf-8") as f:
                    idf = json.load(f)
            except Exception as exc:  # noqa: BLE001
                logger.warning("IDF not loaded for %s: %s", self._model_name_or_path, exc)
                return None

        idf_vec = torch.zeros(tokenizer.vocab_size, dtype=torch.float32)
        for tok, w in idf.items():
            tid = tokenizer._convert_token_to_id_with_added_voc(tok)
            if tid is not None and 0 <= int(tid) < tokenizer.vocab_size:
                idf_vec[int(tid)] = float(w)

        nonzero = int((idf_vec > 0).sum().item())
        logger.info(
            "IDF loaded for %s: %d non-zero entries out of %d vocab tokens. "
            "Query encoding will use inference-free IDF path.",
            self._model_name_or_path,
            nonzero,
            tokenizer.vocab_size,
        )
        return idf_vec

    def _encode_query_idf(self, texts: list[str], is_query: bool) -> EncodeOutput:
        self._check_loaded()
        if self._tokenizer is None or self._idf is None or self._vocab_size is None:
            raise RuntimeError(ERR_NOT_LOADED)

        batch_encoding = self._tokenizer(
            texts,
            max_length=self._max_seq_length,
            truncation=True,
            padding=False,
            return_attention_mask=False,
            return_token_type_ids=False,
        )

        idf = self._idf
        special_ids = set(self._special_token_ids)

        sparse_list: list[SparseVector] = []
        for input_ids in batch_encoding["input_ids"]:
            unique_ids = torch.tensor(
                sorted(tid for tid in set(input_ids) if tid not in special_ids),
                dtype=torch.long,
            )
            if unique_ids.numel() == 0:
                sparse_list.append(
                    SparseVector(
                        indices=np.array([], dtype=np.int32),
                        values=np.array([], dtype=np.float32),
                    )
                )
                continue
            values = idf[unique_ids]
            keep = values > 0
            sparse_list.append(
                SparseVector(
                    indices=unique_ids[keep].numpy().astype(np.int32),
                    values=values[keep].numpy().astype(np.float32),
                )
            )

        return EncodeOutput(sparse=sparse_list, batch_size=len(texts), is_query=is_query)
