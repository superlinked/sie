from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, ClassVar, Literal

import numpy as np
import torch

from sie_server.adapters._cuda_graphs import memory_budget
from sie_server.adapters._flash_base import FlashBaseAdapter
from sie_server.adapters._flash_pack import build_position_ids
from sie_server.adapters._modernbert_flash_graphs import PackedForward, VarlenGraphRunner, unsupported_reason
from sie_server.adapters._spec import AdapterSpec
from sie_server.adapters._transformers_compat import install_transformers_security_fixes
from sie_server.adapters._types import ERR_NOT_LOADED, ERR_REQUIRES_TEXT, ComputePrecision
from sie_server.adapters._utils import apply_rotary_pos_emb
from sie_server.adapters.qwen2_flash import _fused_layers
from sie_server.adapters.qwen2_flash_cross_encoder._prefix import (
    PrefixCache,
    PrefixEntry,
    SharedForward,
    plan_shared_forward,
)
from sie_server.core.inference_output import ScoreOutput
from sie_server.core.oom import is_oom_error
from sie_server.types.inputs import InvalidInputError, InvalidMediaError, Item

if TYPE_CHECKING:
    from pathlib import Path

    from transformers import PreTrainedTokenizerFast

logger = logging.getLogger(__name__)

_ERR_CUDA_REQUIRED = "Qwen2FlashCrossEncoder requires CUDA for Flash Attention."
_ERR_UNSUPPORTED_MEDIA = "Qwen2FlashCrossEncoder accepts text-only query and document items"

# Positions the graphs' rotary tables cover: a cached prefix and its tail must fit.
_GRAPH_WINDOW = 4096
# Tail tokens a graph may hold, at least and at most (the smallest bucket; half the rotary window).
_MIN_GRAPH_TOKENS = 64
_MAX_GRAPH_TOKENS = _GRAPH_WINDOW // 2
# Cached prefixes that may hold graphs at once; the least recently used gives way.
_MAX_PREFIX_GRAPHS = 8
# Instructions and queries whose token ids and counts are remembered across requests.
_MEMO_TEXTS = 512


def default_graph_tokens(hidden_size: int) -> int:
    """Tail tokens a graph over a cached prefix holds, by the model's width.

    A graph pays off while the GPU waits on the host: below the size where
    the forward's GPU time passes the time the host needs to launch its
    kernels. On an L4 that is about 190 tokens for a 2,560-wide, 36-layer
    model (Qwen3-Reranker-4B) and several hundred for a 1,024-wide one. Past
    it, padding a forward to its bucket costs more than the graph saves. A
    faster GPU moves the point up: ``cuda_graph_tokens`` overrides this.
    """
    return 512 if hidden_size <= 1024 else 128


def _remember(memo: dict[Any, Any], key: Any, value: Any) -> None:
    """Keep ``value`` in a memo of at most ``_MEMO_TEXTS`` entries, the oldest first out."""
    memo[key] = value
    while len(memo) > _MEMO_TEXTS:
        memo.pop(next(iter(memo)), None)


class _PrefixGraphRunner(VarlenGraphRunner):
    """The runner of one cached prefix: together the prefixes' graphs hold what one model's graphs may."""

    @staticmethod
    def _memory_budget(device: torch.device) -> int:
        return memory_budget(device) // _MAX_PREFIX_GRAPHS


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
    'and the Instruct provided. Note that the answer can only be "yes" or "no".'
)
QWEN3_DEFAULT_INSTRUCTION = "Given a web search query, retrieve relevant passages that answer the query"
# The model card tokenizes "<Instruct>: {i}\n<Query>: {q}\n<Document>: {d}" as one
# string, so the space after "<Document>:" joins the document's first word (one
# " License" token, never " " then "License"). The pre-tokenizer always splits
# after that colon, so a row keeps the reference's ids when the prefix ends at
# "<Document>:" and the document is tokenized with the space in front of it.
QWEN3_DOCUMENT_SEPARATOR = " "

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

    Three load-time options, each off unless a profile turns it on:

    * ``fused_kernels`` runs the layers' elementwise work as the embedding
      sibling's Triton kernels (``qwen2_flash._fused_layers``), bit for bit
      equal to the eager layers.
    * ``prefix_reuse`` (``qwen3`` format) encodes the tokens before the
      document once for the rows that share them, and keeps a repeated
      prefix's keys and values across requests in a cache of at most
      ``prefix_cache_mb`` (see ``_prefix``).
    * ``cuda_graphs: bucketed`` replays a one-row forward over a cached prefix
      as a CUDA graph, where the GPU otherwise waits on the host that launches
      its kernels.
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
            "_score_weight",
            "_score_bias",
            "_chat_prefix_ids",
            "_chat_suffix_ids",
            "_task_prompt_ids",
            "_sep_ids",
            "_prefix_cache",
            "_prefix_graphs",
            "_rope_tables",
        ),
    )

    def __init__(
        self,
        model_name_or_path: str | Path,
        *,
        trust_remote_code: bool = False,
        max_seq_length: int = 8192,
        compute_precision: ComputePrecision = "bfloat16",
        revision: str | None = None,
        yes_token: str = "1",  # noqa: S107
        no_token: str = "0",  # noqa: S107
        input_format: InputFormat = "mxbai",
        score_mode: ScoreMode = "logit_diff",
        default_instruction: str | None = None,
        fused_kernels: bool = False,
        prefix_reuse: bool = False,
        prefix_cache_mb: int = 512,
        cuda_graphs: str | bool = "off",
        cuda_graph_tokens: int | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the adapter.

        Args:
            model_name_or_path: HuggingFace model ID or local path.
            trust_remote_code: Whether to trust remote code.
            max_seq_length: Maximum sequence length for query+document.
            compute_precision: Compute precision (bfloat16 recommended).
            revision: Optional HuggingFace revision/branch/commit SHA to pin when
                loading model artifacts. Forwarded to ``from_pretrained(..., revision=...)``.
            yes_token: Text whose first token ID is the "positive" logit.
            no_token: Text whose first token ID is the "negative" logit.
            input_format: Template preset — ``"mxbai"`` or ``"qwen3"``.
            score_mode: ``"logit_diff"`` (yes − no) or ``"log_softmax"``
                (probability via log-softmax then exp).
            default_instruction: Instruction text used by the ``qwen3``
                template when no per-request instruction is supplied.
            fused_kernels: If True, run each layer's residual add + RMSNorm,
                q/k RMSNorm, rotary embedding and SiLU-gated MLP product as
                single Triton kernels. Scores equal the eager layers'.
            prefix_reuse: If True (``qwen3`` format), rows that start with the
                same tokens (template, instruction and query) attend to one
                encoding of them, and a prefix seen twice is kept across
                requests. Scores move by floating-point rounding.
            prefix_cache_mb: Device memory the cross-request prefix cache may
                hold, in MiB; 0 shares a prefix inside a forward only.
            cuda_graphs: "off" (the default) or "bucketed": whether a one-row
                forward over a cached prefix replays as a CUDA graph (see
                ``sie_server.adapters._modernbert_flash_graphs`` for the
                runner). Needs ``prefix_reuse``. An operator setting, fixed
                at load.
            cuda_graph_tokens: The longest tail (document and closing tokens)
                a graph holds, 64 to 2048; longer ones run eagerly. Defaults
                by the model's width (``default_graph_tokens``), sized for an
                L4; a faster GPU gains from more.
            **kwargs: Additional arguments (ignored).

        Raises:
            ValueError: If ``cuda_graphs`` is not "off" or "bucketed",
                ``prefix_cache_mb`` is negative, or ``cuda_graph_tokens`` is
                outside 64 to 2048.
        """
        _ = kwargs
        if cuda_graphs is False:  # YAML reads an unquoted ``off`` as False
            cuda_graphs = "off"
        if cuda_graphs not in ("off", "bucketed"):
            msg = f"Qwen2FlashCrossEncoder cuda_graphs must be 'off' or 'bucketed', got {cuda_graphs!r}"
            raise ValueError(msg)
        if isinstance(prefix_cache_mb, bool) or not isinstance(prefix_cache_mb, int) or prefix_cache_mb < 0:
            msg = f"Qwen2FlashCrossEncoder prefix_cache_mb must be a non-negative integer, got {prefix_cache_mb!r}"
            raise ValueError(msg)
        if cuda_graph_tokens is not None and (
            isinstance(cuda_graph_tokens, bool)
            or not isinstance(cuda_graph_tokens, int)
            or not _MIN_GRAPH_TOKENS <= cuda_graph_tokens <= _MAX_GRAPH_TOKENS
        ):
            msg = (
                f"Qwen2FlashCrossEncoder cuda_graph_tokens must be an integer from {_MIN_GRAPH_TOKENS} to "
                f"{_MAX_GRAPH_TOKENS}, got {cuda_graph_tokens!r}"
            )
            raise ValueError(msg)
        self._model_name_or_path = str(model_name_or_path)
        self._trust_remote_code = trust_remote_code
        self._max_seq_length = max_seq_length
        self._compute_precision = compute_precision
        self._revision = revision

        # Configurable template / scoring params
        self._yes_token: str = yes_token
        self._no_token: str = no_token
        self._input_format: InputFormat = input_format
        self._score_mode: ScoreMode = score_mode
        self._default_instruction: str | None = default_instruction
        self._fused_kernels = fused_kernels
        self._prefix_reuse = prefix_reuse
        self._prefix_cache_mb = prefix_cache_mb
        self._cuda_graphs: str = cuda_graphs
        self._cuda_graph_tokens = cuda_graph_tokens

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
        self._score_weight: torch.Tensor | None = None
        self._score_bias: torch.Tensor | None = None

        # Pre-tokenized templates (set during load)
        self._chat_prefix_ids: list[int] = []
        self._chat_suffix_ids: list[int] = []
        self._task_prompt_ids: list[int] = []
        self._sep_ids: list[int] = []
        self._document_separator_ids: list[int] = []

        # Shared prefixes (set during load when ``prefix_reuse`` applies)
        self._prefix_cache: PrefixCache | None = None
        # One graph runner per cached prefix, dropped with it.
        self._prefix_graphs: dict[int, tuple[PrefixEntry, VarlenGraphRunner]] | None = None
        self._graph_tokens = 0
        # With ``prefix_reuse``: the tokens of recent (instruction, query) prefixes
        # and the token counts of recent caller texts, so a repeated question is
        # not tokenized again on every request.
        self._prefix_ids_memo: dict[tuple[str, str], list[int]] = {}
        self._text_tokens_memo: dict[str, int] = {}
        self._rope_tables: tuple[torch.Tensor, torch.Tensor] | None = None

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

        shared_kwargs: dict[str, Any] = {"trust_remote_code": self._trust_remote_code}
        if self._revision is not None:
            shared_kwargs["revision"] = self._revision

        # Load tokenizer
        install_transformers_security_fixes()
        self._tokenizer = AutoTokenizer.from_pretrained(
            self._model_name_or_path,
            **shared_kwargs,
        )

        # Load model with eager attention - we handle flash attention manually
        self._model = AutoModelForCausalLM.from_pretrained(
            self._model_name_or_path,
            torch_dtype=self._dtype,
            attn_implementation="eager",
            **shared_kwargs,
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

        # Reranking consumes only the configured negative/positive token
        # logits. Cache those two output-head rows, in float32, so each
        # request avoids a full-vocabulary projection and a per-call cast.
        score_token_ids = torch.tensor(
            [self._no_token_id, self._yes_token_id],
            dtype=torch.long,
            device=device,
        )
        lm_head = self._model.lm_head
        self._score_weight = lm_head.weight.index_select(0, score_token_ids).detach().float()
        self._score_bias = (
            lm_head.bias.index_select(0, score_token_ids).detach().float() if lm_head.bias is not None else None
        )

        # Pre-tokenize templates based on input format
        self._pre_tokenize_templates()
        self._resolve_serving_options(device)

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

    def _resolve_serving_options(self, device: str) -> None:
        """Turn off the load-time options this model or device cannot run, and build what the others need."""
        if self._fused_kernels and not _fused_layers.supported(self._model.model):
            logger.warning(
                "fused_kernels needs Triton and Qwen2/Qwen3 layers with a SiLU-gated MLP; "
                "%s runs the eager layers instead",
                self._model_name_or_path,
            )
            self._fused_kernels = False
        if self._prefix_reuse and self._input_format != "qwen3":
            logger.warning(
                "prefix_reuse applies to the qwen3 input format; %s (%s) scores every row whole",
                self._model_name_or_path,
                self._input_format,
            )
            self._prefix_reuse = False
        self._prefix_cache = None
        self._prefix_graphs = None
        self._rope_tables = None
        self._prefix_ids_memo = {}
        self._text_tokens_memo = {}
        if not self._prefix_reuse:
            if self._cuda_graphs != "off":
                logger.warning(
                    "cuda_graphs replays forwards over a cached prefix and needs prefix_reuse; %s runs eagerly",
                    self._model_name_or_path,
                )
            return
        if self._prefix_cache_mb:
            self._prefix_cache = PrefixCache(self._prefix_cache_mb * 2**20)
        if self._cuda_graphs == "off" or self._prefix_cache is None:
            return
        reason = unsupported_reason(device)
        if reason is not None:
            logger.warning(
                "cuda_graphs=%s does not apply to %s, which runs eagerly: %s",
                self._cuda_graphs,
                self._model_name_or_path,
                reason,
            )
            return
        self._prefix_graphs = {}
        self._graph_tokens = self._cuda_graph_tokens or default_graph_tokens(self._hidden_size)

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
            self._document_separator_ids = enc(QWEN3_DOCUMENT_SEPARATOR, add_special_tokens=False)
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
            logits: ``[batch_size, 2]`` tensor ordered as ``[no, yes]``.

        Returns:
            [batch_size] float32 scores.
        """
        # Float32 throughout: in bfloat16, P(yes) has a step of 2**-8 near 1.0,
        # so every candidate above about 0.996 ties at exactly 1.0 and the
        # ranking among the most relevant candidates falls back to input order.
        logits = logits.float()
        no_logits = logits[:, 0]
        yes_logits = logits[:, 1]

        if self._score_mode == "log_softmax":
            # Stack [no, yes] and apply log-softmax, take P(yes)
            pair = torch.stack([no_logits, yes_logits], dim=-1)  # [B, 2]
            log_probs = torch.nn.functional.log_softmax(pair, dim=-1)
            return log_probs[:, 1].exp()

        # logit_diff (default)
        return yes_logits - no_logits

    def _project_score_logits(self, last_hidden: torch.Tensor) -> torch.Tensor:
        """Project last-token states onto only the configured score tokens, in float32.

        The two score rows are cached in float32 at load. The yes/no logits are
        the whole score, so they are not rounded to bfloat16.
        """
        if self._score_weight is None:
            raise RuntimeError(ERR_NOT_LOADED)
        return torch.nn.functional.linear(last_hidden.float(), self._score_weight, self._score_bias)

    def score(
        self,
        query: Item,
        items: list[Item],
        *,
        instruction: str | None = None,
        options: dict[str, Any] | None = None,
    ) -> list[float]:
        """Score items against a query through the batched pair path."""
        self._check_loaded()
        if not items:
            return []
        output = self.score_pairs(
            [query] * len(items),
            items,
            instruction=instruction,
            options=options,
        )
        return output.scores.tolist()

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
        if len(queries) != len(docs):
            msg = f"score_pairs requires equal-length queries and docs, got {len(queries)} queries and {len(docs)} docs"
            raise ValueError(msg)

        opts = options or {}
        max_length = self._runtime_max_length(opts.get("max_seq_length"))

        if not queries:
            return ScoreOutput(scores=np.empty(0, dtype=np.float32))

        # Build each row with the chat template, as the tokens before the
        # document (qwen3 format) and the rest; the row is their concatenation.
        rows: list[tuple[list[int], list[int]]] = []
        content_counts: list[int] | None = [] if self._input_format == "qwen3" else None
        caller_text_tokens: dict[str, int] = {}
        user_prefixes: dict[str, list[int]] = {}
        texts = [
            (self._extract_text_only(query_item), self._extract_text_only(doc_item))
            for query_item, doc_item in zip(queries, docs, strict=True)
        ]
        # Every document in one batched call: the fast tokenizer encodes them in
        # parallel and returns, per text, exactly the ids a call on it alone returns.
        doc_ids_list: list[list[int] | None] = [None] * len(texts)
        if content_counts is not None:
            doc_ids_list = self._tokenizer(
                [QWEN3_DOCUMENT_SEPARATOR + doc_text for _, doc_text in texts],
                add_special_tokens=False,
                return_attention_mask=False,
            )["input_ids"]
        for (query_text, doc_text), doc_ids in zip(texts, doc_ids_list, strict=True):
            if content_counts is None:
                input_ids = self._build_input_ids(
                    query_text,
                    doc_text,
                    max_length=max_length,
                    instruction=instruction,
                )
                rows.append(([], input_ids))
            else:
                prefix_ids, tail_ids, document_tokens = self._split_qwen3_pair(
                    query_text,
                    doc_text,
                    max_length=max_length,
                    instruction=instruction,
                    user_prefixes=user_prefixes,
                    doc_ids=doc_ids,
                )
                caller_tokens = self._caller_text_tokens(query_text, caller_text_tokens)
                if instruction:
                    caller_tokens += self._caller_text_tokens(instruction, caller_text_tokens)
                content_counts.append(min(len(prefix_ids) + len(tail_ids), caller_tokens + document_tokens))
                rows.append((prefix_ids, tail_ids))

        seq_lengths = [len(prefix) + len(tail) for prefix, tail in rows]

        with torch.inference_mode():
            logits = self._forward_shared(rows) if self._prefix_reuse else None
            if logits is None:
                logits = self._forward_rows(rows, seq_lengths)
            scores_tensor = self._compute_scores(logits)
            scores_array = scores_tensor.cpu().numpy().astype(np.float32)

        return ScoreOutput(
            scores=scores_array,
            input_token_counts=seq_lengths,
            content_token_counts=content_counts,
        )

    def _forward_rows(self, rows: list[tuple[list[int], list[int]]], seq_lengths: list[int]) -> torch.Tensor:
        """Score-token logits of whole rows, each its own sequence of one packed forward."""
        total_tokens = sum(seq_lengths)
        max_seqlen = max(seq_lengths)
        batch_size = len(rows)

        # Pack input_ids
        input_ids_packed = torch.tensor(
            [tok for prefix, tail in rows for ids in (prefix, tail) for tok in ids],
            dtype=torch.long,
            device=self._device,
        )

        # Build cu_seqlens
        cu_seqlens = torch.zeros(batch_size + 1, dtype=torch.int32, device=self._device)
        cu_seqlens[1:] = torch.cumsum(
            torch.tensor(seq_lengths, dtype=torch.int32, device=self._device),
            dim=0,
        )
        position_ids = self._build_position_ids(cu_seqlens, batch_size)
        return self._forward_flash(
            input_ids_packed,
            cu_seqlens,
            max_seqlen,
            total_tokens,
            batch_size,
            position_ids,
        )

    def _caller_text_tokens(self, text: str, cache: dict[str, int]) -> int:
        """Token count of one caller-supplied string tokenized on its own."""
        count = cache.get(text)
        if count is None:
            count = self._text_tokens_memo.get(text) if self._prefix_reuse else None
            if count is None:
                assert self._tokenizer is not None
                count = len(self._tokenizer.encode(text, add_special_tokens=False))
                if self._prefix_reuse:
                    _remember(self._text_tokens_memo, text, count)
            cache[text] = count
        return count

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
        if max_doc_len < 0:
            raise InvalidInputError(
                f"max_seq_length={effective_max_length} is too small for the reranker query and template "
                f"({predefined_len} tokens before the document)"
            )
        if max_doc_len == 0:
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
        input_ids, _document_tokens = self._build_qwen3_pair(
            query,
            document,
            max_length=max_length,
            instruction=instruction,
        )
        return input_ids

    def _build_qwen3_pair(
        self,
        query: str,
        document: str,
        *,
        max_length: int | None = None,
        instruction: str | None = None,
    ) -> tuple[list[int], int]:
        """Return the Qwen3 pair's input IDs and its post-truncation document token count."""
        prefix_ids, tail_ids, document_tokens = self._split_qwen3_pair(
            query,
            document,
            max_length=max_length,
            instruction=instruction,
        )
        return prefix_ids + tail_ids, document_tokens

    def _split_qwen3_pair(
        self,
        query: str,
        document: str,
        *,
        max_length: int | None = None,
        instruction: str | None = None,
        user_prefixes: dict[str, list[int]] | None = None,
        doc_ids: list[int] | None = None,
    ) -> tuple[list[int], list[int], int]:
        """The Qwen3 pair as the tokens before its document and the rest, with the document's token count.

        The row is ``prefix + tail``. The prefix (chat template, instruction
        and query) is the same list for every pair of one query when the
        caller passes ``user_prefixes``, a memo for one ``score_pairs`` call.
        ``doc_ids`` are the token ids of ``QWEN3_DOCUMENT_SEPARATOR + document``
        when the caller has them. The prefix ends at "<Document>:" and the
        separating space is tokenized with the document, so the row is the ids
        the model card's single tokenization of the whole user text gives.
        """
        assert self._tokenizer is not None
        effective_max_length = max_length or self._max_seq_length
        prefix_ids = user_prefixes.get(query) if user_prefixes is not None else None
        if prefix_ids is None:
            inst = instruction or self._default_instruction or QWEN3_DEFAULT_INSTRUCTION
            prefix_ids = self._prefix_ids_memo.get((inst, query)) if self._prefix_reuse else None
            if prefix_ids is None:
                user_prefix = f"<Instruct>: {inst}\n<Query>: {query}\n<Document>:"
                prefix_ids = self._chat_prefix_ids + self._tokenizer.encode(
                    user_prefix,
                    add_special_tokens=False,
                )
                if self._prefix_reuse:
                    _remember(self._prefix_ids_memo, (inst, query), prefix_ids)
            if user_prefixes is not None:
                user_prefixes[query] = prefix_ids
        if doc_ids is None:
            doc_ids = self._tokenizer.encode(QWEN3_DOCUMENT_SEPARATOR + document, add_special_tokens=False)

        # Truncate document if needed (keep instruction + query intact)
        predefined_len = len(prefix_ids) + len(self._chat_suffix_ids)
        max_doc_len = effective_max_length - predefined_len
        if max_doc_len < 0:
            raise InvalidInputError(
                f"max_seq_length={effective_max_length} is too small for the reranker instruction, query, "
                f"and template ({predefined_len} tokens before the document)"
            )
        if max_doc_len == 0:
            doc_ids = []
        elif len(doc_ids) > max_doc_len:
            doc_ids = doc_ids[:max_doc_len]

        # The separator is template, not caller text: where it stays a token of
        # its own (before a digit or a bracket, say) it is not a document token.
        separator = self._document_separator_ids
        document_tokens = len(doc_ids)
        if separator and doc_ids[: len(separator)] == separator:
            document_tokens -= len(separator)
        return prefix_ids, doc_ids + self._chat_suffix_ids, document_tokens

    def _extract_text_only(self, item: Item) -> str:
        if item.images or item.audio is not None or item.video is not None or item.document is not None:
            raise InvalidMediaError(_ERR_UNSUPPORTED_MEDIA)
        if item.text is None or not item.text.strip():
            raise InvalidInputError(ERR_REQUIRES_TEXT.format(adapter_name="Qwen2FlashCrossEncoder"))
        return item.text

    def _runtime_max_length(self, value: Any) -> int:
        """Validate a request override and clamp it to the load-time ceiling."""
        if value is None:
            return self._max_seq_length
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise InvalidInputError("options.max_seq_length must be a positive integer")
        return min(value, self._max_seq_length)

    # ------------------------------------------------------------------
    # Position encoding
    # ------------------------------------------------------------------

    def _build_position_ids(self, cu_seqlens: torch.Tensor, batch_size: int) -> torch.Tensor:
        """Build position IDs for packed sequences."""
        _ = batch_size
        return build_position_ids(cu_seqlens)

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
            Score-token logits ``[batch_size, 2]`` ordered as ``[no, yes]``.
        """
        # Get embeddings (Qwen2 uses only token embeddings, RoPE applied in attention)
        hidden = self._model.model.embed_tokens(input_ids)
        cos, sin = self._compute_rope(self._rotary_emb(), position_ids)
        hidden = self._run_layers(hidden, cu_seqlens, max_seqlen, total_tokens, cos, sin)

        # Final layer norm
        hidden = self._model.model.norm(hidden)

        # Extract last token hidden state for each sequence
        last_indices = (cu_seqlens[1:] - 1).long()  # Last token of each sequence
        last_hidden = hidden[last_indices]  # [batch_size, hidden_size]

        return self._project_score_logits(last_hidden)

    def _rotary_emb(self) -> Any:
        """The model's rotary embedding (Qwen3 stores it at model level, Qwen2 per layer)."""
        base = self._model.model
        if hasattr(base, "rotary_emb"):
            return base.rotary_emb
        return base.layers[0].self_attn.rotary_emb

    def _run_layers(
        self,
        hidden: torch.Tensor,
        cu_seqlens: torch.Tensor,
        max_seqlen: int,
        total_tokens: int,
        cos: torch.Tensor,
        sin: torch.Tensor,
        *,
        attend: _fused_layers.AttendFn | None = None,
    ) -> torch.Tensor:
        """Run the decoder layers over the packed ``hidden``; the residual stream before the final norm.

        With ``fused_kernels`` the elementwise work runs as the embedding
        sibling's Triton kernels, bit for bit equal to the loop below. With
        ``attend``, each layer's attention is that call (a shared-prefix
        forward) in place of causal varlen flash attention over ``cu_seqlens``.
        """
        if self._fused_kernels:
            return _fused_layers.run_layers(
                self._model.model, hidden, cu_seqlens, max_seqlen, total_tokens, cos, sin, causal=True, attend=attend
            )

        softmax_scale = 1.0 / (self._head_dim**0.5)

        # Run transformer layers
        for index, layer in enumerate(self._model.model.layers):
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

            if attend is None:
                from flash_attn import flash_attn_varlen_func

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
            else:
                attn_out = attend(index, query, key, value)
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

        return hidden

    # ------------------------------------------------------------------
    # Shared prefixes (see ``_prefix``)
    # ------------------------------------------------------------------

    def _forward_shared(self, rows: list[tuple[list[int], list[int]]]) -> torch.Tensor | None:
        """Score-token logits through a forward that shares prefixes; None when no row can share one."""
        cache = self._prefix_cache
        try:
            plan = plan_shared_forward(rows, cache, self._prefix_token_bytes())
            if plan is None:
                return None
            if len(rows) == 1 and len(plan.cached) == 1 and not plan.store:
                logits = self._replay_cached(plan.cached[0], rows[0][1])
                if logits is not None:
                    return logits
            return self._run_shared(plan)
        except Exception as exc:
            # Cached prefixes and their graphs hold device memory: release it
            # so the caller's out-of-memory recovery has it to work with.
            if is_oom_error(exc):
                self._drop_prefixes()
            raise

    def _prefix_token_bytes(self) -> int:
        """Device bytes one cached prefix token holds: its keys and values in every layer."""
        dtype = self._dtype or self._resolve_dtype()
        layers = len(self._model.model.layers)
        return 2 * layers * self._num_kv_heads * self._head_dim * dtype.itemsize

    def _drop_prefixes(self) -> None:
        if self._prefix_cache is not None:
            self._prefix_cache.clear()
        if self._prefix_graphs is not None:
            for _, runner in self._prefix_graphs.values():
                runner.clear()
            self._prefix_graphs.clear()

    def _prefix_attention(self, cu_seqlens_q: torch.Tensor, cu_seqlens_k: torch.Tensor, max_q: int, max_k: int) -> Any:
        """Causal varlen flash attention of tails over ``prefix + tail`` keys: ``(q, k, v) -> output``.

        With more keys than queries in a sequence, flash-attn aligns the
        causal mask to the bottom right: a tail token attends to the whole
        prefix and to the tail up to itself.
        """
        from flash_attn import flash_attn_varlen_func

        softmax_scale = 1.0 / (self._head_dim**0.5)

        def attention(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
            return flash_attn_varlen_func(
                query,
                key,
                value,
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_k=cu_seqlens_k,
                max_seqlen_q=max_q,
                max_seqlen_k=max_k,
                causal=True,
                softmax_scale=softmax_scale,
            )

        return attention

    def _run_shared(self, plan: SharedForward) -> torch.Tensor:
        """Run a shared-prefix forward eagerly; score-token logits ``[rows, 2]`` ordered as ``[no, yes]``."""
        device = self._device
        base = self._model.model
        input_ids = torch.from_numpy(plan.input_ids).to(device)
        positions = torch.from_numpy(plan.positions).to(device)
        cu_seqlens_q = torch.from_numpy(plan.cu_seqlens_q).to(device)
        cu_seqlens_k = torch.from_numpy(plan.cu_seqlens_k).to(device)
        k_index = torch.from_numpy(plan.k_index).to(device) if plan.k_index is not None else None
        total_tokens = len(plan.input_ids)

        hidden = base.embed_tokens(input_ids)
        cos, sin = self._compute_rope(self._rotary_emb(), positions)

        # The prefixes this forward keeps: filled layer by layer, cached once whole.
        def empty(tokens: int) -> torch.Tensor:
            shape = (len(base.layers), tokens, self._num_kv_heads, self._head_dim)
            return torch.empty(shape, dtype=hidden.dtype, device=device)

        new = [
            (key, start, end, PrefixEntry(keys=empty(end - start), values=empty(end - start)))
            for key, start, end in plan.store
        ]
        cached = plan.cached
        attention = self._prefix_attention(cu_seqlens_q, cu_seqlens_k, plan.max_seqlen_q, plan.max_seqlen_k)

        def attend(index: int, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
            for _, start, end, entry in new:
                entry.keys[index].copy_(key[start:end])
                entry.values[index].copy_(value[start:end])
            if cached:
                key = torch.cat([*(entry.keys[index] for entry in cached), key])
                value = torch.cat([*(entry.values[index] for entry in cached), value])
            if k_index is not None:
                key = key.index_select(0, k_index)
                value = value.index_select(0, k_index)
            return attention(query, key, value)

        hidden = self._run_layers(hidden, cu_seqlens_q, plan.max_seqlen_q, total_tokens, cos, sin, attend=attend)
        last_hidden = hidden[torch.from_numpy(plan.last_index).to(device)]
        logits = self._project_score_logits(base.norm(last_hidden))
        if self._prefix_cache is not None and new:
            for key, _, _, entry in new:
                self._prefix_cache.put(key, entry)
            self._drop_evicted_prefix_graphs()
            logger.info(
                "Kept %d repeated prefix(es) of %s for reuse across requests: %d cached, %d MB",
                len(new),
                self._model_name_or_path,
                len(self._prefix_cache),
                self._prefix_cache.held_bytes // 2**20,
            )
        return logits

    # ------------------------------------------------------------------
    # CUDA graphs over a cached prefix
    # ------------------------------------------------------------------

    def _replay_cached(self, entry: PrefixEntry, tail: list[int]) -> torch.Tensor | None:
        """Score-token logits of one row over a cached prefix from a replayed CUDA graph; None to run it eagerly."""
        graphs = self._prefix_graphs
        if graphs is None or len(tail) > self._graph_tokens or entry.tokens + self._graph_tokens > _GRAPH_WINDOW:
            return None
        held = graphs.pop(id(entry), None)
        if held is None:
            self._drop_evicted_prefix_graphs()
            while len(graphs) >= _MAX_PREFIX_GRAPHS:  # the least recently used gives way
                graphs.pop(next(iter(graphs)))[1].clear()
            held = (entry, self._prefix_graph_runner(entry))
        graphs[id(entry)] = held  # the most recently used last

        def head(packed: PackedForward) -> torch.Tensor:
            last = (packed.cu_seqlens[1:] - 1).long()
            return self._project_score_logits(packed.hidden[last])

        runner = held[1]
        recorded = runner.stats.recorded
        logits = runner.run(tail, [len(tail)], head)
        if runner.stats.recorded != recorded:
            logger.info(
                "CUDA graph for %d tail tokens over a cached prefix of %d recorded for %s: "
                "%d graphs on this prefix, %d MB; %d prefixes hold graphs",
                len(tail),
                entry.tokens,
                self._model_name_or_path,
                runner.graph_count,
                runner.held_bytes // 2**20,
                len(graphs),
            )
        return logits

    def _drop_evicted_prefix_graphs(self) -> None:
        """Drop the graphs of prefixes the cache no longer holds, with the keys and values they read."""
        graphs, cache = self._prefix_graphs, self._prefix_cache
        if not graphs or cache is None:
            return
        live = cache.entry_ids()
        for key in [key for key in graphs if key not in live]:
            graphs.pop(key)[1].clear()

    def _prefix_graph_runner(self, entry: PrefixEntry) -> VarlenGraphRunner:
        """A runner whose graphs score one tail over ``entry``: the tail's forward up to the final norm.

        The graph reads the cached keys and values in place, so it lives no
        longer than the cache holds ``entry``. A graph has slots for several
        sequences; this one is offered one row, which fills the first slot,
        and the other slots stay empty.
        """
        base = self._model.model
        if self._rope_tables is None:
            self._rope_tables = self._compute_rope(self._rotary_emb(), torch.arange(_GRAPH_WINDOW, device=self._device))
        cos_table, sin_table = self._rope_tables
        prefix_tokens = entry.tokens

        def encode(
            input_ids: torch.Tensor, positions: torch.Tensor, cu_seqlens: torch.Tensor, max_seqlen: int, total: int
        ) -> torch.Tensor:
            hidden = base.embed_tokens(input_ids)
            at = (positions + prefix_tokens).long()
            # The row's keys follow the prefix's; the empty slots hold neither.
            cu_seqlens_k = cu_seqlens + (cu_seqlens > 0).to(cu_seqlens.dtype) * prefix_tokens
            attention = self._prefix_attention(cu_seqlens, cu_seqlens_k, max_seqlen, prefix_tokens + max_seqlen)

            def attend(index: int, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
                return attention(query, torch.cat([entry.keys[index], key]), torch.cat([entry.values[index], value]))

            hidden = self._run_layers(
                hidden, cu_seqlens, max_seqlen, total, cos_table[at], sin_table[at], attend=attend
            )
            return base.norm(hidden)

        return _PrefixGraphRunner(
            encode,
            hidden_size=self._hidden_size,
            dtype=self._dtype or self._resolve_dtype(),
            device=self._device or "cuda",
            window=self._graph_tokens,
            pad_token_id=getattr(self._tokenizer, "pad_token_id", None) or 0,
            max_tokens=self._graph_tokens,
            name=self._model_name_or_path,
        )

    def _resolve_dtype(self) -> torch.dtype:
        """Resolve compute dtype."""
        dtype_map = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }
        return dtype_map.get(self._compute_precision, torch.bfloat16)
