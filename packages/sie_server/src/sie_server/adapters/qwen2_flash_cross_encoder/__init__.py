from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, ClassVar

import torch

from sie_server.adapters._flash_base import FlashBaseAdapter
from sie_server.adapters._spec import AdapterSpec
from sie_server.adapters._types import ERR_NOT_LOADED, ERR_REQUIRES_TEXT, ComputePrecision
from sie_server.adapters._utils import apply_rotary_pos_emb, extract_text
from sie_server.core.inference_output import ScoreOutput
from sie_server.types.inputs import Item

if TYPE_CHECKING:
    from pathlib import Path

    from transformers import PreTrainedTokenizerFast

logger = logging.getLogger(__name__)

_ERR_CUDA_REQUIRED = "Qwen2FlashCrossEncoder requires CUDA for Flash Attention."

# Chat template for mxbai-rerank models
_CHAT_PREFIX = "<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\n<|im_start|>user\n"
_CHAT_SUFFIX = "<|im_end|>\n<|im_start|>assistant\n"
_TASK_PROMPT = "You are a search relevance expert who evaluates how well documents match search queries. For each query-document pair, carefully analyze the semantic relationship between them, then provide your binary relevance judgment (0 for not relevant, 1 for relevant).\nRelevance:"


class Qwen2FlashCrossEncoderAdapter(FlashBaseAdapter):
    """Cross-encoder adapter for Qwen2-based causal LM rerankers.

    Uses flash_attn_varlen_func for variable-length sequences without padding.
    Implements the mxbai-rerank scoring mechanism with logit differences.
    """

    fallback_adapter_path: ClassVar[str | None] = "cross_encoder:CrossEncoderAdapter"
    fallback_kwargs_overrides: ClassVar[dict[str, Any]] = {"attn_implementation": "sdpa"}

    spec = AdapterSpec(
        inputs=("text",),
        outputs=("score",),
        unload_fields=(
            "_model",
            "_tokenizer",
            "_dtype",
            "_num_heads",
            "_num_kv_heads",
            "_head_dim",
            "_hidden_size",
            "_yes_token_id",
            "_no_token_id",
            "_chat_prefix_ids",
            "_chat_suffix_ids",
            "_task_prompt_ids",
            "_sep_ids",
        ),
    )

    def __init__(
        self,
        model_name_or_path: str | Path,
        *,
        trust_remote_code: bool = False,
        max_seq_length: int = 8192,
        compute_precision: ComputePrecision = "bfloat16",
        **kwargs: Any,
    ) -> None:
        """Initialize the adapter.

        Args:
            model_name_or_path: HuggingFace model ID or local path.
            trust_remote_code: Whether to trust remote code.
            max_seq_length: Maximum sequence length for query+document.
            compute_precision: Compute precision (bfloat16 recommended).
            **kwargs: Additional arguments (ignored).
        """
        _ = kwargs
        self._model_name_or_path = str(model_name_or_path)
        self._trust_remote_code = trust_remote_code
        self._max_seq_length = max_seq_length
        self._compute_precision = compute_precision

        # Loaded state
        self._model: Any = None
        self._tokenizer: PreTrainedTokenizerFast | None = None
        self._device: str | None = None
        self._dtype: torch.dtype | None = None

        # Model config (set during load)
        self._num_heads: int = 0
        self._num_kv_heads: int = 0
        self._head_dim: int = 0
        self._hidden_size: int = 0

        # Token IDs for scoring (set during load)
        self._yes_token_id: int = 0  # Token for "1"
        self._no_token_id: int = 0  # Token for "0"

        # Pre-tokenized templates (set during load)
        self._chat_prefix_ids: list[int] = []
        self._chat_suffix_ids: list[int] = []
        self._task_prompt_ids: list[int] = []
        self._sep_ids: list[int] = []

    def load(self, device: str) -> None:
        """Load model weights onto the specified device."""
        if not device.startswith("cuda"):
            raise RuntimeError(_ERR_CUDA_REQUIRED)

        from transformers import AutoModelForCausalLM, AutoTokenizer

        self._device = device
        self._dtype = self._resolve_dtype()

        logger.info(
            "Loading %s with Flash Attention varlen (dtype=%s)",
            self._model_name_or_path,
            self._dtype,
        )

        # Load tokenizer
        self._tokenizer = AutoTokenizer.from_pretrained(
            self._model_name_or_path,
            trust_remote_code=self._trust_remote_code,
        )

        # Load model with eager attention - we handle flash attention manually
        self._model = AutoModelForCausalLM.from_pretrained(
            self._model_name_or_path,
            torch_dtype=self._dtype,
            attn_implementation="eager",
            trust_remote_code=self._trust_remote_code,
        )
        self._model.to(device)
        self._model.eval()

        # Cache config values
        config = self._model.config
        self._num_heads = config.num_attention_heads
        self._num_kv_heads = config.num_key_value_heads
        self._hidden_size = config.hidden_size
        self._head_dim = self._hidden_size // self._num_heads

        # Get token IDs for scoring
        self._yes_token_id = self._tokenizer.encode("1", add_special_tokens=False)[0]
        self._no_token_id = self._tokenizer.encode("0", add_special_tokens=False)[0]

        # Pre-tokenize templates
        self._chat_prefix_ids = self._tokenizer.encode(_CHAT_PREFIX, add_special_tokens=False)
        self._chat_suffix_ids = self._tokenizer.encode(_CHAT_SUFFIX, add_special_tokens=False)
        self._task_prompt_ids = self._tokenizer.encode(_TASK_PROMPT, add_special_tokens=False)
        self._sep_ids = self._tokenizer.encode("\n", add_special_tokens=False)

        logger.info(
            "Loaded: hidden=%d, heads=%d, kv_heads=%d, layers=%d, yes_tok=%d, no_tok=%d",
            self._hidden_size,
            self._num_heads,
            self._num_kv_heads,
            config.num_hidden_layers,
            self._yes_token_id,
            self._no_token_id,
        )

    def _resolve_dtype(self) -> torch.dtype:
        """Resolve compute dtype."""
        dtype_map = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }
        return dtype_map.get(self._compute_precision, torch.bfloat16)

    def score(
        self,
        query: Item,
        items: list[Item],
        *,
        instruction: str | None = None,
        options: dict[str, Any] | None = None,
    ) -> list[float]:
        """Score items against a query using flash attention varlen.

        Args:
            query: Query item (must have text).
            items: Items to score against the query.
            instruction: Optional instruction (currently unused for mxbai format).

        Returns:
            List of relevance scores (higher = more relevant).
        """
        self._check_loaded()

        query_text = extract_text(query, err_msg=ERR_REQUIRES_TEXT.format(adapter_name="Qwen2FlashCrossEncoder"))

        # Build input sequences with chat template
        all_input_ids = []
        for item in items:
            doc_text = extract_text(item, err_msg=ERR_REQUIRES_TEXT.format(adapter_name="Qwen2FlashCrossEncoder"))
            input_ids = self._build_input_ids(query_text, doc_text)
            all_input_ids.append(input_ids)

        # Build packed representation
        seq_lengths = [len(ids) for ids in all_input_ids]
        total_tokens = sum(seq_lengths)
        max_seqlen = max(seq_lengths)
        batch_size = len(items)

        # Pack input_ids
        input_ids_packed = torch.tensor(
            [tok for ids in all_input_ids for tok in ids],
            dtype=torch.long,
            device=self._device,
        )

        # Build cu_seqlens
        cu_seqlens = torch.zeros(batch_size + 1, dtype=torch.int32, device=self._device)
        for i, length in enumerate(seq_lengths):
            cu_seqlens[i + 1] = cu_seqlens[i] + length

        with torch.inference_mode():
            # Build position IDs
            position_ids = self._build_position_ids(cu_seqlens, batch_size)

            # Run forward pass with flash attention
            logits = self._forward_flash(
                input_ids_packed,
                cu_seqlens,
                max_seqlen,
                total_tokens,
                batch_size,
                position_ids,
            )

            # Score = logit("1") - logit("0") at last position of each sequence
            yes_logits = logits[:, self._yes_token_id]
            no_logits = logits[:, self._no_token_id]
            scores_tensor = (yes_logits - no_logits).float()

            scores = scores_tensor.cpu().tolist()

        return scores

    def score_pairs(
        self,
        queries: list[Item],
        docs: list[Item],
        *,
        instruction: str | None = None,
        options: dict[str, Any] | None = None,
    ) -> ScoreOutput:
        """Score (query, doc) pairs in a batch.

        Batched version of score() for cross-request batching.

        Args:
            queries: Query items (parallel to docs).
            docs: Document items to score.
            instruction: Optional instruction (currently unused for mxbai format).
            options: Runtime options (config defaults -> profile -> request overrides).

        Returns:
            ScoreOutput containing scores for each (query, doc) pair.
        """
        self._check_loaded()
        if self._tokenizer is None:
            raise RuntimeError(ERR_NOT_LOADED)

        opts = options or {}
        max_length = opts.get("max_seq_length", self._max_seq_length)

        # Build input sequences with chat template
        all_input_ids = []
        for query, doc in zip(queries, docs, strict=True):
            query_text = extract_text(query, err_msg=ERR_REQUIRES_TEXT.format(adapter_name="Qwen2FlashCrossEncoder"))
            doc_text = extract_text(doc, err_msg=ERR_REQUIRES_TEXT.format(adapter_name="Qwen2FlashCrossEncoder"))
            input_ids = self._build_input_ids(query_text, doc_text, max_length=max_length)
            all_input_ids.append(input_ids)

        # Build packed representation
        seq_lengths = [len(ids) for ids in all_input_ids]
        total_tokens = sum(seq_lengths)
        max_seqlen = max(seq_lengths)
        batch_size = len(queries)

        # Pack input_ids
        input_ids_packed = torch.tensor(
            [tok for ids in all_input_ids for tok in ids],
            dtype=torch.long,
            device=self._device,
        )

        # Build cu_seqlens
        cu_seqlens = torch.zeros(batch_size + 1, dtype=torch.int32, device=self._device)
        for i, length in enumerate(seq_lengths):
            cu_seqlens[i + 1] = cu_seqlens[i] + length

        with torch.inference_mode():
            # Build position IDs
            position_ids = self._build_position_ids(cu_seqlens, batch_size)

            # Run forward pass with flash attention
            logits = self._forward_flash(
                input_ids_packed,
                cu_seqlens,
                max_seqlen,
                total_tokens,
                batch_size,
                position_ids,
            )

            # Score = logit("1") - logit("0") at last position of each sequence
            yes_logits = logits[:, self._yes_token_id]
            no_logits = logits[:, self._no_token_id]
            scores_tensor = (yes_logits - no_logits).float()

            # Convert to float32 numpy array and wrap in ScoreOutput
            import numpy as np

            scores_array = scores_tensor.cpu().numpy().astype(np.float32)

        return ScoreOutput(scores=scores_array)

    def _build_input_ids(self, query: str, document: str, *, max_length: int | None = None) -> list[int]:
        r"""Build input IDs with chat template.

        Format:
        <chat_prefix>query: {query}\ndocument: {document}\n<task_prompt><chat_suffix>

        Args:
            query: Query text.
            document: Document text.
            max_length: Maximum sequence length override (defaults to self._max_seq_length).
        """
        effective_max_length = max_length if max_length is not None else self._max_seq_length

        # Tokenize query and document
        query_prompt = f"query: {query}"
        doc_prompt = f"document: {document}"

        query_ids = self._tokenizer.encode(query_prompt, add_special_tokens=False)
        doc_ids = self._tokenizer.encode(doc_prompt, add_special_tokens=False)

        # Truncate document if needed (keep query intact)
        predefined_len = (
            len(self._chat_prefix_ids)
            + len(query_ids)
            + len(self._sep_ids)
            + len(self._sep_ids)
            + len(self._task_prompt_ids)
            + len(self._chat_suffix_ids)
        )
        max_doc_len = effective_max_length - predefined_len
        if len(doc_ids) > max_doc_len:
            doc_ids = doc_ids[:max_doc_len]

        # Build full sequence
        input_ids = (
            self._chat_prefix_ids
            + query_ids
            + self._sep_ids
            + doc_ids
            + self._sep_ids
            + self._task_prompt_ids
            + self._chat_suffix_ids
        )

        return input_ids

    def _build_position_ids(self, cu_seqlens: torch.Tensor, batch_size: int) -> torch.Tensor:
        """Build position IDs for packed sequences."""
        pos_list = []
        for i in range(batch_size):
            seq_len = int(cu_seqlens[i + 1].item() - cu_seqlens[i].item())
            pos_list.append(torch.arange(0, seq_len, device=self._device))
        return torch.cat(pos_list)

    def _compute_rope(
        self,
        rotary_emb: Any,
        position_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute RoPE cos/sin values for packed positions.

        Qwen2RotaryEmbedding.forward(x, position_ids) returns (cos, sin).
        - x is used only for dtype/device (shape doesn't matter)
        - position_ids shape: [batch, seq] or [1, total_tokens] for packed
        - Returns: (cos, sin) each [batch, seq, head_dim]
        """
        dtype = self._dtype

        # Create dummy x for dtype/device reference
        dummy_x = torch.zeros(1, 1, 1, self._head_dim, device=self._device, dtype=dtype)

        # Position IDs need to be [1, total_tokens] for packed sequences
        pos_ids = position_ids.unsqueeze(0)  # [1, total_tokens]

        cos, sin = rotary_emb(dummy_x, pos_ids)

        # Squeeze batch dimension and return [total_tokens, head_dim]
        cos = cos.squeeze(0).to(dtype)  # [total_tokens, head_dim]
        sin = sin.squeeze(0).to(dtype)  # [total_tokens, head_dim]

        return cos, sin

    def _forward_flash(
        self,
        input_ids: torch.Tensor,
        cu_seqlens: torch.Tensor,
        max_seqlen: int,
        total_tokens: int,
        batch_size: int,
        position_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Run forward pass with flash attention varlen.

        Returns:
            Logits tensor [batch_size, vocab_size] for last token of each sequence.
        """
        from flash_attn import flash_attn_varlen_func

        # Get embeddings (Qwen2 uses only token embeddings, RoPE applied in attention)
        hidden = self._model.model.embed_tokens(input_ids)

        softmax_scale = 1.0 / (self._head_dim**0.5)

        # Precompute RoPE using model-level rotary_emb (Qwen2 stores it at model level)
        rotary_emb = self._model.model.rotary_emb
        cos, sin = self._compute_rope(rotary_emb, position_ids)

        # Run transformer layers
        for layer in self._model.model.layers:
            attn = layer.self_attn

            # Pre-norm (Qwen2 uses RMSNorm before attention)
            normed_hidden = layer.input_layernorm(hidden)

            # Separate Q, K, V projections
            query = attn.q_proj(normed_hidden)
            key = attn.k_proj(normed_hidden)
            value = attn.v_proj(normed_hidden)

            # Reshape for attention
            query = query.view(total_tokens, self._num_heads, self._head_dim)
            key = key.view(total_tokens, self._num_kv_heads, self._head_dim)
            value = value.view(total_tokens, self._num_kv_heads, self._head_dim)

            # Apply RoPE
            query, key = apply_rotary_pos_emb(query, key, cos, sin)

            # Flash attention with causal masking (decoder)
            attn_out = flash_attn_varlen_func(
                query,
                key,
                value,
                cu_seqlens_q=cu_seqlens,
                cu_seqlens_k=cu_seqlens,
                max_seqlen_q=max_seqlen,
                max_seqlen_k=max_seqlen,
                causal=True,  # Causal for decoder
                softmax_scale=softmax_scale,
            )
            attn_out = attn_out.reshape(total_tokens, self._hidden_size)

            # Output projection
            attn_out = attn.o_proj(attn_out)

            # Residual connection
            hidden = hidden + attn_out

            # Pre-norm for MLP
            normed_hidden = layer.post_attention_layernorm(hidden)

            # MLP
            mlp_out = layer.mlp(normed_hidden)

            # Residual connection
            hidden = hidden + mlp_out

        # Final layer norm
        hidden = self._model.model.norm(hidden)

        # Extract last token hidden state for each sequence
        last_indices = (cu_seqlens[1:] - 1).long()  # Last token of each sequence
        last_hidden = hidden[last_indices]  # [batch_size, hidden_size]

        # Get logits via lm_head
        logits = self._model.lm_head(last_hidden)  # [batch_size, vocab_size]

        return logits
