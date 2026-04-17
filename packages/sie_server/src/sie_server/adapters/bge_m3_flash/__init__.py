from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar

import numpy as np
import torch
from huggingface_hub import snapshot_download
from torch import nn
from torch.nn import functional

from sie_server.adapters._flash_base import FlashBaseAdapter
from sie_server.adapters._spec import AdapterSpec
from sie_server.adapters._types import ERR_NOT_LOADED, ComputePrecision
from sie_server.adapters._utils import validate_output_types
from sie_server.adapters.peft_lora_mixin import PEFTLoRAMixin
from sie_server.core.inference_output import EncodeOutput, SparseVector
from sie_server.types.inputs import Item

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerFast, XLMRobertaModel

logger = logging.getLogger(__name__)

_ERR_CPU_NOT_SUPPORTED = "BGEM3FlashAdapter requires CUDA. Use bge_m3 adapter for CPU."


class BGEM3FlashAdapter(PEFTLoRAMixin, FlashBaseAdapter):
    """BGE-M3 adapter using Flash Attention 2 with variable-length sequences.

    This adapter eliminates padding waste by packing sequences and using
    flash_attn_varlen_func. Achieves higher throughput than SDPA-based adapter.
    """

    DENSE_DIM = 1024
    SPARSE_DIM = 250002
    MULTIVECTOR_DIM = 1024

    fallback_adapter_path: ClassVar[str | None] = "bge_m3:BGEM3Adapter"

    spec = AdapterSpec(
        inputs=("text",),
        outputs=("dense", "sparse", "multivector"),
        dense_dim=1024,
        sparse_dim=250002,
        multivector_dim=1024,
        unload_fields=("_model", "_tokenizer", "_colbert_linear", "_sparse_linear"),
    )

    def __init__(
        self,
        model_name_or_path: str | Path = "BAAI/bge-m3",
        *,
        normalize: bool = True,
        max_seq_length: int = 8192,
        compute_precision: ComputePrecision = "float16",
        **kwargs: Any,
    ) -> None:
        """Initialize the adapter.

        Args:
            model_name_or_path: HuggingFace model ID or local path.
            normalize: Whether to L2-normalize dense embeddings.
            max_seq_length: Maximum sequence length (default 8192).
            compute_precision: Compute precision (float16 recommended for flash).
            **kwargs: Additional arguments (ignored, for compatibility).
        """
        _ = kwargs
        self._model_name_or_path = str(model_name_or_path)
        self._normalize = normalize
        self._max_seq_length = max_seq_length
        self._compute_precision = compute_precision

        self._model: XLMRobertaModel | None = None
        self._tokenizer: PreTrainedTokenizerFast | None = None
        self._colbert_linear: nn.Linear | None = None
        self._sparse_linear: nn.Linear | None = None
        self._device: str | None = None
        self._padding_idx: int = 1  # XLMRoberta default, set properly in load()

    def load(self, device: str) -> None:
        """Load the model onto the specified device.

        Args:
            device: Device string (must be "cuda" or "cuda:X").

        Raises:
            RuntimeError: If device is not CUDA (flash attention requires GPU).
        """
        if not device.startswith("cuda"):
            raise RuntimeError(_ERR_CPU_NOT_SUPPORTED)

        from transformers import AutoModel, AutoTokenizer

        self._device = device
        dtype = self._resolve_dtype()

        logger.info(
            "Loading %s on device=%s with dtype=%s, attn=flash_varlen",
            self._model_name_or_path,
            device,
            dtype,
        )

        self._tokenizer = AutoTokenizer.from_pretrained(self._model_name_or_path)

        # Load model with eager attention - we'll run our own flash attention
        self._model = AutoModel.from_pretrained(
            self._model_name_or_path,
            torch_dtype=dtype,
            attn_implementation="eager",  # We handle attention manually
        )
        self._model.to(device)
        self._model.eval()

        # Get padding_idx for XLMRoberta position ID calculation
        self._padding_idx = self._model.embeddings.padding_idx
        logger.debug("XLMRoberta padding_idx: %d", self._padding_idx)

        self._load_linear_layers(self._model_name_or_path, dtype, device)

    def _load_linear_layers(self, model_path: str, dtype: torch.dtype, device: str) -> None:
        """Load the colbert and sparse linear layers from checkpoint."""
        hidden_size = self._model.config.hidden_size  # type: ignore[union-attr]

        # Resolve the actual directory: could be a local path or HF model ID
        base_path = Path(model_path)
        if not base_path.is_dir():
            base_path = Path(snapshot_download(model_path))

        colbert_path = base_path / "colbert_linear.pt"
        if colbert_path.exists():
            self._colbert_linear = nn.Linear(hidden_size, hidden_size)
            state_dict = torch.load(colbert_path, map_location=device, weights_only=True)
            self._colbert_linear.load_state_dict(state_dict)
            self._colbert_linear.to(device=device, dtype=dtype)
            self._colbert_linear.eval()
        else:
            logger.warning("colbert_linear.pt not found at %s", base_path)

        sparse_path = base_path / "sparse_linear.pt"
        if sparse_path.exists():
            self._sparse_linear = nn.Linear(hidden_size, 1)
            state_dict = torch.load(sparse_path, map_location=device, weights_only=True)
            self._sparse_linear.load_state_dict(state_dict)
            self._sparse_linear.to(device=device, dtype=dtype)
            self._sparse_linear.eval()
        else:
            logger.warning("sparse_linear.pt not found at %s", base_path)

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
            output_types: Which outputs to compute ("dense", "sparse", "multivector").
            instruction: Optional instruction prefix.
            is_query: Whether items are queries (affects instruction handling).
            prepared_items: Not used by this adapter.

        Returns:
            EncodeOutput with requested embedding types.

        Note:
            LoRA is handled via set_active_lora() called by the worker before encode().
        """
        self._check_loaded()
        if self._tokenizer is None:
            raise RuntimeError(ERR_NOT_LOADED)

        validate_output_types(output_types, {"dense", "sparse", "multivector"}, "BGEM3FlashAdapter")

        # Resolve runtime options (config defaults -> profile -> request overrides)
        opts = options or {}
        normalize = opts.get("normalize", self._normalize)

        texts = self._extract_texts(items, instruction)

        # Tokenize each sequence individually (no padding)
        encodings = [
            self._tokenizer(
                text,
                max_length=self._max_seq_length,
                truncation=True,
                return_tensors="pt",
            )
            for text in texts
        ]

        # Build packed representation
        seq_lengths = [enc["input_ids"].shape[1] for enc in encodings]
        total_tokens = sum(seq_lengths)
        max_seqlen = max(seq_lengths)

        # Pack input_ids
        input_ids_packed = torch.cat([enc["input_ids"].squeeze(0) for enc in encodings]).to(self._device)

        # Build cu_seqlens (cumulative sequence lengths)
        cu_seqlens = torch.zeros(len(texts) + 1, dtype=torch.int32, device=self._device)
        for i, length in enumerate(seq_lengths):
            cu_seqlens[i + 1] = cu_seqlens[i] + length

        with torch.inference_mode():
            # Build XLMRoberta-style position IDs (start at padding_idx + 1)
            position_ids_packed = self._build_position_ids(cu_seqlens, len(texts))

            # Run embeddings
            hidden = self._run_embeddings(input_ids_packed, position_ids_packed)

            # Run transformer layers with flash attention
            hidden = self._run_transformer_flash(hidden, cu_seqlens, max_seqlen, total_tokens)

            # Compute requested embeddings
            results = self._compute_embeddings(
                hidden,
                input_ids_packed,
                cu_seqlens,
                seq_lengths,
                output_types,
                normalize=normalize,
            )

        return self._to_inference_output(results, output_types, len(items), is_query)

    def _build_position_ids(self, cu_seqlens: torch.Tensor, num_seqs: int) -> torch.Tensor:
        """Build XLMRoberta-style position IDs for packed sequences.

        XLMRoberta uses position IDs starting at padding_idx + 1, not 0.
        This is critical for matching the padded model's output.
        """
        pos_list = []
        for i in range(num_seqs):
            seq_len = cu_seqlens[i + 1].item() - cu_seqlens[i].item()
            # XLMRoberta positions: [padding_idx+1, padding_idx+2, ..., padding_idx+seq_len]
            pos_list.append(
                torch.arange(
                    self._padding_idx + 1,
                    self._padding_idx + 1 + seq_len,
                    device=self._device,
                )
            )
        return torch.cat(pos_list)

    def _run_embeddings(self, input_ids: torch.Tensor, position_ids: torch.Tensor) -> torch.Tensor:
        """Compute embeddings for packed input."""
        embeddings = self._model.embeddings  # type: ignore[union-attr]

        word_emb = embeddings.word_embeddings(input_ids)
        pos_emb = embeddings.position_embeddings(position_ids)
        token_type_emb = embeddings.token_type_embeddings(torch.zeros_like(input_ids))

        hidden = word_emb + pos_emb + token_type_emb
        hidden = embeddings.LayerNorm(hidden)
        hidden = embeddings.dropout(hidden)

        return hidden

    def _run_transformer_flash(
        self,
        hidden: torch.Tensor,
        cu_seqlens: torch.Tensor,
        max_seqlen: int,
        total_tokens: int,
    ) -> torch.Tensor:
        """Run transformer layers using flash_attn_varlen_func."""
        from flash_attn import flash_attn_varlen_func

        num_heads = self._model.config.num_attention_heads  # type: ignore[union-attr]
        hidden_size = self._model.config.hidden_size  # type: ignore[union-attr]
        head_dim = hidden_size // num_heads
        softmax_scale = 1.0 / (head_dim**0.5)

        for layer in self._model.encoder.layer:  # type: ignore[union-attr]
            attention = layer.attention.self

            # QKV projections
            query = attention.query(hidden).view(total_tokens, num_heads, head_dim)
            key = attention.key(hidden).view(total_tokens, num_heads, head_dim)
            value = attention.value(hidden).view(total_tokens, num_heads, head_dim)

            # Flash attention with variable-length sequences
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
            attn_out = attn_out.reshape(total_tokens, hidden_size)

            # Output projection and residual
            attn_out = layer.attention.output.dense(attn_out)
            attn_out = layer.attention.output.dropout(attn_out)
            hidden = layer.attention.output.LayerNorm(attn_out + hidden)

            # FFN
            inter = layer.intermediate.dense(hidden)
            inter = layer.intermediate.intermediate_act_fn(inter)
            out = layer.output.dense(inter)
            out = layer.output.dropout(out)
            hidden = layer.output.LayerNorm(out + hidden)

        return hidden

    def _compute_embeddings(
        self,
        hidden: torch.Tensor,
        input_ids: torch.Tensor,
        cu_seqlens: torch.Tensor,
        seq_lengths: list[int],
        output_types: list[str],
        *,
        normalize: bool | None = None,
    ) -> dict[str, Any]:
        """Compute requested embeddings from the model output."""
        normalize = normalize if normalize is not None else self._normalize
        results: dict[str, Any] = {}
        num_seqs = len(seq_lengths)

        if "dense" in output_types:
            # Extract CLS token from each sequence
            cls_embeddings = []
            for i in range(num_seqs):
                start = cu_seqlens[i].item()
                cls_embeddings.append(hidden[start])
            dense_vecs = torch.stack(cls_embeddings)
            if normalize:
                dense_vecs = functional.normalize(dense_vecs, p=2, dim=-1)
            results["dense"] = dense_vecs

        if "sparse" in output_types and self._sparse_linear is not None:
            token_weights = torch.relu(self._sparse_linear(hidden)).squeeze(-1)
            results["sparse"] = self._compute_sparse_weights(token_weights, input_ids, cu_seqlens, seq_lengths)

        if "multivector" in output_types and self._colbert_linear is not None:
            results["multivector"] = self._compute_multivector(hidden, cu_seqlens, seq_lengths, normalize=normalize)

        return results

    def _compute_sparse_weights(
        self,
        token_weights: torch.Tensor,
        input_ids: torch.Tensor,
        cu_seqlens: torch.Tensor,
        seq_lengths: list[int],
    ) -> list[dict[int, float]]:
        """Compute sparse lexical weights per item."""
        special_tokens = {
            self._tokenizer.cls_token_id,
            self._tokenizer.eos_token_id,
            self._tokenizer.pad_token_id,
            self._tokenizer.unk_token_id,
        }
        special_tokens.discard(None)

        results = []
        for i in range(len(seq_lengths)):
            start = cu_seqlens[i].item()
            end = cu_seqlens[i + 1].item()

            weights = token_weights[start:end].float().cpu().numpy()
            ids = input_ids[start:end].cpu().numpy()

            sparse_dict: dict[int, float] = {}
            for tid, weight in zip(ids, weights, strict=True):
                if tid in special_tokens or weight <= 0:
                    continue
                tid_int = int(tid)
                if tid_int not in sparse_dict or weight > sparse_dict[tid_int]:
                    sparse_dict[tid_int] = float(weight)

            results.append(sparse_dict)

        return results

    def _compute_multivector(
        self,
        hidden: torch.Tensor,
        cu_seqlens: torch.Tensor,
        seq_lengths: list[int],
        *,
        normalize: bool = True,
    ) -> list[torch.Tensor]:
        """Compute ColBERT-style multi-vector embeddings."""
        results = []
        for i in range(len(seq_lengths)):
            start = cu_seqlens[i].item()
            end = cu_seqlens[i + 1].item()

            # Skip CLS token (index 0 of each sequence)
            seq_hidden = hidden[start + 1 : end]
            colbert_vecs = self._colbert_linear(seq_hidden)

            if normalize:
                colbert_vecs = functional.normalize(colbert_vecs, p=2, dim=-1)

            results.append(colbert_vecs)

        return results

    def _extract_texts(self, items: list[Item], instruction: str | None) -> list[str]:
        """Extract texts from items, optionally prepending instruction."""
        texts = []
        for item in items:
            if item.text is None:
                raise ValueError("BGEM3FlashAdapter requires text input")
            text = item.text
            if instruction is not None:
                text = f"{instruction} {text}"
            texts.append(text)
        return texts

    def _to_inference_output(
        self,
        embeddings: dict[str, Any],
        output_types: list[str],
        batch_size: int,
        is_query: bool,
    ) -> EncodeOutput:
        """Convert internal results to EncodeOutput format."""
        dense_np: np.ndarray | None = None
        sparse_list: list[SparseVector] | None = None
        multivector_list: list[np.ndarray] | None = None

        if "dense" in output_types and "dense" in embeddings:
            dense_np = embeddings["dense"].float().cpu().numpy()

        if "sparse" in output_types and "sparse" in embeddings:
            sparse_list = []
            for weights in embeddings["sparse"]:
                if weights:
                    indices = np.array(list(weights.keys()), dtype=np.int32)
                    values = np.array(list(weights.values()), dtype=np.float32)
                else:
                    indices = np.array([], dtype=np.int32)
                    values = np.array([], dtype=np.float32)
                sparse_list.append(SparseVector(indices=indices, values=values))

        if "multivector" in output_types and "multivector" in embeddings:
            multivector_list = [vecs.float().cpu().numpy() for vecs in embeddings["multivector"]]

        return EncodeOutput(
            dense=dense_np,
            sparse=sparse_list,
            multivector=multivector_list,
            batch_size=batch_size,
            is_query=is_query,
            dense_dim=self.DENSE_DIM if dense_np is not None else None,
            multivector_token_dim=self.MULTIVECTOR_DIM if multivector_list else None,
        )
