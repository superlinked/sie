from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, ClassVar, Literal

import numpy as np
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

# ---------------------------------------------------------------------------
# Input-format presets
# ---------------------------------------------------------------------------

InputFormat = Literal["mxbai", "qwen3"]
ScoreMode = Literal["logit_diff", "log_softmax"]

# mxbai-rerank template pieces
_MXBAI_SYSTEM = "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."
_MXBAI_TASK_PROMPT = (
    "You are a search relevance expert who evaluates how well documents match "
    "search queries. For each query-document pair, carefully analyze the "
    "semantic relationship between them, then provide your binary relevance "
    "judgment (0 for not relevant, 1 for relevant).\nRelevance:"
)

# Qwen3-Reranker template pieces
_QWEN3_SYSTEM = (
    "Judge whether the Document meets the requirements based on the Query "
    "and the Instruct provided. Note that the answer can only be 'yes' or 'no'."
)
QWEN3_DEFAULT_INSTRUCTION = "Given a web search query, retrieve relevant passages that answer the query"

# Pre-composed chat template strings (used by the eval wrapper for non-flash inference)
QWEN3_CHAT_PREFIX = f"<|im_start|>system\n{_QWEN3_SYSTEM}<|im_end|>\n<|im_start|>user\n"
QWEN3_CHAT_SUFFIX = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"


class Qwen2FlashCrossEncoderAdapter(FlashBaseAdapter):
    """Cross-encoder adapter for Qwen2/Qwen3-based causal LM rerankers.

    Uses flash_attn_varlen_func for variable-length sequences without padding.
    Supports two input-format presets:

    * ``mxbai`` (default) — mxbai-rerank-v2 models, ``"1"``/``"0"`` tokens,
      raw logit-difference scoring.
    * ``qwen3`` — Qwen3-Reranker models, ``"yes"``/``"no"`` tokens,
      log-softmax probability scoring, ``<think>`` suffix.
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
        yes_token: str = "1",  # noqa: S107
        no_token: str = "0",  # noqa: S107
        input_format: InputFormat = "mxbai",
        score_mode: ScoreMode = "logit_diff",
        default_instruction: str | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the adapter.

        Args:
            model_name_or_path: HuggingFace model ID or local path.
            trust_remote_code: Whether to trust remote code.
            max_seq_length: Maximum sequence length for query+document.
            compute_precision: Compute precision (bfloat16 recommended).
            yes_token: Text whose first token ID is the "positive" logit.
            no_token: Text whose first token ID is the "negative" logit.
            input_format: Template preset — ``"mxbai"`` or ``"qwen3"``.
            score_mode: ``"logit_diff"`` (yes − no) or ``"log_softmax"``
                (probability via log-softmax then exp).
            default_instruction: Instruction text used by the ``qwen3``
                template when no per-request instruction is supplied.
            **kwargs: Additional arguments (ignored).
        """
        _ = kwargs
        self._model_name_or_path = str(model_name_or_path)
        self._trust_remote_code = trust_remote_code
        self._max_seq_length = max_seq_length
        self._compute_precision = compute_precision

        # Configurable template / scoring params
        self._yes_token: str = yes_token
        self._no_token: str = no_token
        self._input_format: InputFormat = input_format
        self._score_mode: ScoreMode = score_mode
        self._default_instruction: str | None = default_instruction

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
        self._yes_token_id: int = 0
        self._no_token_id: int = 0

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
            "Loading %s with Flash Attention varlen (dtype=%s, format=%s, score=%s)",
            self._model_name_or_path,
            self._dtype,
            self._input_format,
            self._score_mode,
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
        # Qwen3 sets head_dim explicitly (e.g. 128) which differs from
        # hidden_size // num_heads.  Fall back for Qwen2 compatibility.
        self._head_dim = getattr(config, "head_dim", self._hidden_size // self._num_heads)

        # Get token IDs for scoring using configurable tokens
        self._yes_token_id = self._tokenizer.encode(
            self._yes_token,
            add_special_tokens=False,
        )[0]
        self._no_token_id = self._tokenizer.encode(
            self._no_token,
            add_special_tokens=False,
        )[0]

        # Pre-tokenize templates based on input format
        self._pre_tokenize_templates()

        logger.info(
            "Loaded: hidden=%d, heads=%d, kv_heads=%d, layers=%d, yes_tok=%d ('%s'), no_tok=%d ('%s')",
            self._hidden_size,
            self._num_heads,
            self._num_kv_heads,
            config.num_hidden_layers,
            self._yes_token_id,
            self._yes_token,
            self._no_token_id,
            self._no_token,
        )

        # Clamp configured max_seq_length to whatever the tokenizer/model
        # actually support to avoid OOB position embeddings on long inputs.
        self._max_seq_length = self._resolve_tokenizer_ceiling(
            self._tokenizer,
            self._model,
            self._max_seq_length,
        )

    # ------------------------------------------------------------------
    # Template pre-tokenization
    # ------------------------------------------------------------------

    def _pre_tokenize_templates(self) -> None:
        """Pre-tokenize chat template pieces based on input_format."""
        assert self._tokenizer is not None
        enc = self._tokenizer.encode

        if self._input_format == "qwen3":
            prefix = f"<|im_start|>system\n{_QWEN3_SYSTEM}<|im_end|>\n<|im_start|>user\n"
            suffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
            self._chat_prefix_ids = enc(prefix, add_special_tokens=False)
            self._chat_suffix_ids = enc(suffix, add_special_tokens=False)
            # qwen3 format puts instruction/query/document inline — no
            # separate task prompt
            self._task_prompt_ids = []
        else:
            # mxbai (default)
            prefix = f"<|im_start|>system\n{_MXBAI_SYSTEM}<|im_end|>\n<|im_start|>user\n"
            suffix = "<|im_end|>\n<|im_start|>assistant\n"
            self._chat_prefix_ids = enc(prefix, add_special_tokens=False)
            self._chat_suffix_ids = enc(suffix, add_special_tokens=False)
            self._task_prompt_ids = enc(
                _MXBAI_TASK_PROMPT,
                add_special_tokens=False,
            )

        self._sep_ids = enc("\n", add_special_tokens=False)

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------

    def _compute_scores(self, logits: torch.Tensor) -> torch.Tensor:
        """Convert raw logits to scores based on score_mode.

        Args:
            logits: [batch_size, vocab_size] tensor from last token.

        Returns:
            [batch_size] float32 scores.
        """
        yes_logits = logits[:, self._yes_token_id]
        no_logits = logits[:, self._no_token_id]

        if self._score_mode == "log_softmax":
            # Stack [no, yes] and apply log-softmax, take P(yes)
            pair = torch.stack([no_logits, yes_logits], dim=-1)  # [B, 2]
            log_probs = torch.nn.functional.log_softmax(pair, dim=-1)
            return log_probs[:, 1].exp().float()

        # logit_diff (default)
        return (yes_logits - no_logits).float()

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
            instruction: Optional instruction (used by qwen3 format).
            options: Runtime options.

        Returns:
            List of relevance scores (higher = more relevant).
        """
        self._check_loaded()

        opts = options or {}
        # Hard-clamp to the load-time ceiling so runtime overrides cannot
        # push past the model's positional capacity. Malformed overrides
        # (None, strings, negatives) fall back to the ceiling.
        max_length = self._coerce_runtime_max_length(opts.get("max_seq_length"), self._max_seq_length)

        query_text = extract_text(
            query,
            err_msg=ERR_REQUIRES_TEXT.format(adapter_name="Qwen2FlashCrossEncoder"),
        )

        if not items:
            return []

        # Build input sequences with chat template
        all_input_ids = []
        for item in items:
            doc_text = extract_text(
                item,
                err_msg=ERR_REQUIRES_TEXT.format(
                    adapter_name="Qwen2FlashCrossEncoder",
                ),
            )
            input_ids = self._build_input_ids(
                query_text,
                doc_text,
                max_length=max_length,
                instruction=instruction,
            )
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
        cu_seqlens = torch.zeros(
            batch_size + 1,
            dtype=torch.int32,
            device=self._device,
        )
        for i, length in enumerate(seq_lengths):
            cu_seqlens[i + 1] = cu_seqlens[i] + length

        with torch.inference_mode():
            position_ids = self._build_position_ids(cu_seqlens, batch_size)

            logits = self._forward_flash(
                input_ids_packed,
                cu_seqlens,
                max_seqlen,
                total_tokens,
                batch_size,
                position_ids,
            )

            scores_tensor = self._compute_scores(logits)
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
            instruction: Optional instruction (used by qwen3 format).
            options: Runtime options (config defaults -> profile -> request overrides).

        Returns:
            ScoreOutput containing scores for each (query, doc) pair.
        """
        self._check_loaded()
        if self._tokenizer is None:
            raise RuntimeError(ERR_NOT_LOADED)

        opts = options or {}
        # Hard-clamp to the load-time ceiling so runtime overrides cannot
        # push past the model's positional capacity. Malformed overrides
        # (None, strings, negatives) fall back to the ceiling.
        max_length = self._coerce_runtime_max_length(opts.get("max_seq_length"), self._max_seq_length)

        if not queries:
            return ScoreOutput(scores=np.empty(0, dtype=np.float32))

        # Build input sequences with chat template
        all_input_ids = []
        for query_item, doc_item in zip(queries, docs, strict=True):
            query_text = extract_text(
                query_item,
                err_msg=ERR_REQUIRES_TEXT.format(
                    adapter_name="Qwen2FlashCrossEncoder",
                ),
            )
            doc_text = extract_text(
                doc_item,
                err_msg=ERR_REQUIRES_TEXT.format(
                    adapter_name="Qwen2FlashCrossEncoder",
                ),
            )
            input_ids = self._build_input_ids(
                query_text,
                doc_text,
                max_length=max_length,
                instruction=instruction,
            )
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
        cu_seqlens = torch.zeros(
            batch_size + 1,
            dtype=torch.int32,
            device=self._device,
        )
        for i, length in enumerate(seq_lengths):
            cu_seqlens[i + 1] = cu_seqlens[i] + length

        with torch.inference_mode():
            position_ids = self._build_position_ids(cu_seqlens, batch_size)

            logits = self._forward_flash(
                input_ids_packed,
                cu_seqlens,
                max_seqlen,
                total_tokens,
                batch_size,
                position_ids,
            )

            scores_tensor = self._compute_scores(logits)
            scores_array = scores_tensor.cpu().numpy().astype(np.float32)

        return ScoreOutput(scores=scores_array)

    # ------------------------------------------------------------------
    # Input construction
    # ------------------------------------------------------------------

    def _build_input_ids(
        self,
        query: str,
        document: str,
        *,
        max_length: int | None = None,
        instruction: str | None = None,
    ) -> list[int]:
        r"""Build input IDs with chat template.

        Args:
            query: Query text.
            document: Document text.
            max_length: Maximum sequence length override.
            instruction: Optional instruction (used by qwen3 format).
        """
        if self._input_format == "qwen3":
            return self._build_input_ids_qwen3(
                query,
                document,
                max_length=max_length,
                instruction=instruction,
            )
        return self._build_input_ids_mxbai(
            query,
            document,
            max_length=max_length,
        )

    def _build_input_ids_mxbai(
        self,
        query: str,
        document: str,
        *,
        max_length: int | None = None,
    ) -> list[int]:
        r"""Build input IDs with mxbai-rerank chat template.

        Format:
        <chat_prefix>query: {query}\ndocument: {document}\n<task_prompt><chat_suffix>
        """
        assert self._tokenizer is not None
        effective_max_length = max_length or self._max_seq_length

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
        if max_doc_len <= 0:
            doc_ids = []
        elif len(doc_ids) > max_doc_len:
            doc_ids = doc_ids[:max_doc_len]

        return (
            self._chat_prefix_ids
            + query_ids
            + self._sep_ids
            + doc_ids
            + self._sep_ids
            + self._task_prompt_ids
            + self._chat_suffix_ids
        )

    def _build_input_ids_qwen3(
        self,
        query: str,
        document: str,
        *,
        max_length: int | None = None,
        instruction: str | None = None,
    ) -> list[int]:
        r"""Build input IDs with Qwen3-Reranker chat template.

        Format:
        <chat_prefix><Instruct>: {instruction}\n<Query>: {query}\n<Document>: {document}<chat_suffix>
        """
        assert self._tokenizer is not None
        effective_max_length = max_length or self._max_seq_length
        inst = instruction or self._default_instruction or QWEN3_DEFAULT_INSTRUCTION

        user_prefix = f"<Instruct>: {inst}\n<Query>: {query}\n<Document>: "
        user_prefix_ids = self._tokenizer.encode(
            user_prefix,
            add_special_tokens=False,
        )
        doc_ids = self._tokenizer.encode(document, add_special_tokens=False)

        # Truncate document if needed (keep instruction + query intact)
        predefined_len = len(self._chat_prefix_ids) + len(user_prefix_ids) + len(self._chat_suffix_ids)
        max_doc_len = effective_max_length - predefined_len
        if max_doc_len <= 0:
            doc_ids = []
        elif len(doc_ids) > max_doc_len:
            doc_ids = doc_ids[:max_doc_len]

        return self._chat_prefix_ids + user_prefix_ids + doc_ids + self._chat_suffix_ids

    # ------------------------------------------------------------------
    # Position encoding
    # ------------------------------------------------------------------

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

    # ------------------------------------------------------------------
    # Flash-attention forward pass
    # ------------------------------------------------------------------

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

        # Precompute RoPE (Qwen3 stores rotary_emb at model level, Qwen2 per-layer)
        if hasattr(self._model.model, "rotary_emb"):
            rotary_emb = self._model.model.rotary_emb
        else:
            rotary_emb = self._model.model.layers[0].self_attn.rotary_emb
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

            # Qwen3 QK-normalization (per-head RMSNorm, before RoPE)
            if hasattr(attn, "q_norm"):
                query = attn.q_norm(query)
            if hasattr(attn, "k_norm"):
                key = attn.k_norm(key)

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
            attn_out = attn_out.reshape(total_tokens, self._num_heads * self._head_dim)

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

    def _resolve_dtype(self) -> torch.dtype:
        """Resolve compute dtype."""
        dtype_map = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }
        return dtype_map.get(self._compute_precision, torch.bfloat16)
