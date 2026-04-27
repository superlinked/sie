"""Stablebridge Pruner-Highlighter adapter for SIE.

Custom adapter that wraps a frozen cross-encoder (BAAI/bge-reranker-v2-m3)
with a trained PruningHead MLP to produce both reranking scores AND
token-level pruning/highlighting probabilities in a single forward pass.

Architecture:
    Input: [BOS] query [SEP] passage [SEP]
                    |
        BGE-reranker-v2-m3 (FROZEN)
            |                   |
    [BOS] hidden state    Token hidden states
            |                   |
    Rerank score        PruningHead MLP
    (from classifier)   (512-dim intermediate)
                            |
                    Token keep probabilities
                        [0.0 - 1.0]

The adapter supports both score() and score_pairs() primitives.
Score responses are extended with pruning metadata via the ExtractOutput.

See: SOW - SIE Beta Testing - Stablebridge Integration.md
"""

from __future__ import annotations

import gc
import logging
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
import torch.nn as nn
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from sie_server.adapters.base import ModelAdapter, ModelCapabilities, ModelDims
from sie_server.core.inference_output import ExtractOutput, ScoreOutput
from sie_server.core.preprocessor import CharCountPreprocessor
from sie_server.types.inputs import Item

logger = logging.getLogger(__name__)

# Type aliases
ComputePrecision = Literal["float16", "bfloat16", "float32"]

# Error messages
_ERR_NOT_LOADED = "Model not loaded. Call load() first."
_ERR_REQUIRES_TEXT = "StablebridgePrunerAdapter requires text input"

# Default pruning thresholds (from training config)
DEFAULT_PRUNE_THRESHOLD = 0.6
DEFAULT_HIGHLIGHT_THRESHOLD = 0.9


class PruningHead(nn.Module):
    """MLP head that predicts per-token keep/drop probabilities.

    Trained on top of frozen encoder hidden states. Only this head's
    weights come from our trained checkpoint; the encoder is not modified.

    Architecture:
        Linear(hidden_size, intermediate_size) → GELU → Dropout → Linear(intermediate_size, 1) → Sigmoid
    """

    def __init__(
        self,
        hidden_size: int = 1024,
        intermediate_size: int = 512,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, intermediate_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(intermediate_size, 1),
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Predict keep probabilities for each token.

        Args:
            hidden_states: [batch, seq_len, hidden_size]
            attention_mask: [batch, seq_len] - 1 for real tokens, 0 for padding

        Returns:
            [batch, seq_len] probabilities in [0, 1]
        """
        logits = self.classifier(hidden_states).squeeze(-1)  # [batch, seq_len]
        probs = torch.sigmoid(logits)
        if attention_mask is not None:
            probs = probs * attention_mask.float()
        return probs


class StablebridgePrunerAdapter(ModelAdapter):
    """Custom SIE adapter for unified reranking + token-level pruning.

    This adapter loads:
    1. Frozen BAAI/bge-reranker-v2-m3 with output_hidden_states=True
    2. Trained PruningHead MLP weights from HuggingFace

    For each (query, document) pair, it returns:
    - Rerank score (from the base model's classification head)
    - Token-level keep probabilities (from PruningHead)

    The pruning probabilities are returned as metadata in the score response,
    enabling downstream consumers to prune/highlight text.
    """

    def __init__(
        self,
        model_name_or_path: str | Path = "BAAI/bge-reranker-v2-m3",
        *,
        pruning_head_path: str = "sugiv/stablebridge-pruner-highlighter",
        pruning_head_file: str = "best.pt",
        hidden_size: int = 1024,
        intermediate_size: int = 512,
        dropout: float = 0.2,
        max_seq_length: int = 8192,
        compute_precision: ComputePrecision = "bfloat16",
        prune_threshold: float = DEFAULT_PRUNE_THRESHOLD,
        highlight_threshold: float = DEFAULT_HIGHLIGHT_THRESHOLD,
        trust_remote_code: bool = False,
        **kwargs: Any,
    ) -> None:
        """Initialize the adapter.

        Args:
            model_name_or_path: HuggingFace ID for the base cross-encoder.
            pruning_head_path: HuggingFace repo or local path for PruningHead weights.
            pruning_head_file: Filename of the checkpoint within pruning_head_path.
            hidden_size: Encoder hidden dimension (1024 for BGE-reranker-v2-m3).
            intermediate_size: PruningHead intermediate layer size.
            dropout: PruningHead dropout rate.
            max_seq_length: Maximum sequence length for tokenization.
            compute_precision: Model compute precision.
            prune_threshold: Token probability threshold for pruning decisions.
            highlight_threshold: Token probability threshold for highlighting.
            trust_remote_code: Whether to trust remote code.
        """
        _ = kwargs  # Accept extra args for compatibility
        self._model_name_or_path = str(model_name_or_path)
        self._pruning_head_path = pruning_head_path
        self._pruning_head_file = pruning_head_file
        self._hidden_size = hidden_size
        self._intermediate_size = intermediate_size
        self._dropout = dropout
        self._max_seq_length = max_seq_length
        self._compute_precision = compute_precision
        self._prune_threshold = prune_threshold
        self._highlight_threshold = highlight_threshold
        self._trust_remote_code = trust_remote_code

        self._model: AutoModelForSequenceClassification | None = None
        self._pruning_head: PruningHead | None = None
        self._tokenizer: AutoTokenizer | None = None
        self._device: str | None = None

    @property
    def capabilities(self) -> ModelCapabilities:
        """Return model capabilities — score + extract."""
        return ModelCapabilities(
            inputs=["text"],
            outputs=["score", "json"],
        )

    @property
    def dims(self) -> ModelDims:
        """Return dims (cross-encoder, no embeddings)."""
        return ModelDims()

    def load(self, device: str) -> None:
        """Load the base model and pruning head onto device.

        Args:
            device: Target device (e.g., "cuda:0").
        """
        self._device = device

        # Resolve dtype
        dtype_map = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }
        dtype = dtype_map.get(self._compute_precision, torch.bfloat16)

        logger.info(
            "Loading StablebridgePruner: base=%s, head=%s, device=%s, dtype=%s",
            self._model_name_or_path,
            self._pruning_head_path,
            device,
            dtype,
        )

        # 1. Load tokenizer
        self._tokenizer = AutoTokenizer.from_pretrained(
            self._model_name_or_path,
            trust_remote_code=self._trust_remote_code,
        )

        # 2. Load base cross-encoder with hidden states enabled
        self._model = AutoModelForSequenceClassification.from_pretrained(
            self._model_name_or_path,
            torch_dtype=dtype,
            trust_remote_code=self._trust_remote_code,
            output_hidden_states=True,
        )
        self._model.to(device)
        self._model.eval()

        # 3. Load PruningHead MLP
        self._pruning_head = PruningHead(
            hidden_size=self._hidden_size,
            intermediate_size=self._intermediate_size,
            dropout=self._dropout,
        )
        self._load_pruning_head_weights(dtype)
        self._pruning_head.to(device)
        self._pruning_head.eval()

        logger.info("StablebridgePruner loaded successfully")

    def _load_pruning_head_weights(self, dtype: torch.dtype) -> None:
        """Load PruningHead weights from checkpoint.

        Supports both HuggingFace Hub paths and local paths.
        The checkpoint is expected to contain the PruningHead state_dict
        (possibly nested under 'model_state_dict' or 'pruning_head').
        """
        # Resolve path: try huggingface_hub first, then local
        checkpoint_path = self._resolve_checkpoint_path()

        logger.info("Loading PruningHead weights from: %s", checkpoint_path)
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=True)

        # Extract PruningHead state dict from various checkpoint formats
        if isinstance(ckpt, dict):
            if "pruning_head" in ckpt:
                state_dict = ckpt["pruning_head"]
            elif "model_state_dict" in ckpt:
                # Filter to only pruning head keys
                state_dict = {}
                for k, v in ckpt["model_state_dict"].items():
                    if k.startswith("pruning_head."):
                        state_dict[k.replace("pruning_head.", "")] = v
                    elif k.startswith("classifier."):
                        state_dict[k] = v
                if not state_dict:
                    state_dict = ckpt["model_state_dict"]
            elif "state_dict" in ckpt:
                state_dict = ckpt["state_dict"]
            else:
                # Try loading directly — the dict IS the state_dict
                state_dict = ckpt
        else:
            msg = f"Unexpected checkpoint type: {type(ckpt)}"
            raise ValueError(msg)

        self._pruning_head.load_state_dict(state_dict, strict=False)
        self._pruning_head.to(dtype)
        logger.info(
            "PruningHead loaded: %d parameters",
            sum(p.numel() for p in self._pruning_head.parameters()),
        )

    def _resolve_checkpoint_path(self) -> str:
        """Resolve PruningHead checkpoint to a local file path."""
        local_path = Path(self._pruning_head_path)
        if local_path.is_file():
            return str(local_path)
        if (local_path / self._pruning_head_file).is_file():
            return str(local_path / self._pruning_head_file)

        # Try HuggingFace Hub
        try:
            from huggingface_hub import hf_hub_download

            path = hf_hub_download(
                repo_id=self._pruning_head_path,
                filename=self._pruning_head_file,
            )
            return path
        except Exception as e:
            msg = (
                f"Could not find PruningHead weights at '{self._pruning_head_path}' "
                f"(file: '{self._pruning_head_file}'): {e}"
            )
            raise FileNotFoundError(msg) from e

    def unload(self) -> None:
        """Unload models and release GPU memory."""
        device = self._device

        if self._model is not None:
            del self._model
            self._model = None
        if self._pruning_head is not None:
            del self._pruning_head
            self._pruning_head = None
        if self._tokenizer is not None:
            del self._tokenizer
            self._tokenizer = None

        self._device = None

        gc.collect()
        if device and device.startswith("cuda"):
            torch.cuda.empty_cache()
        elif device == "mps":
            torch.mps.empty_cache()

    def score(
        self,
        query: Item,
        items: list[Item],
        *,
        instruction: str | None = None,
        options: dict[str, Any] | None = None,
    ) -> list[float]:
        """Score items against query, returning rerank scores.

        This implements the standard SIE score() interface. For pruning
        metadata, use extract() instead.

        Args:
            query: Query item with text.
            items: Document items to score.
            instruction: Optional instruction prefix.
            options: Runtime options.

        Returns:
            List of rerank scores (higher = more relevant).
        """
        if self._model is None:
            raise RuntimeError(_ERR_NOT_LOADED)

        query_text = self._extract_text(query)
        if instruction:
            query_text = f"{instruction} {query_text}"

        # Tokenize (query, doc) pairs
        pairs = [(query_text, self._extract_text(item)) for item in items]
        inputs = self._tokenize_pairs(pairs)

        # Forward pass — only need classification scores
        with torch.inference_mode():
            outputs = self._model(**inputs)
            logits = outputs.logits.squeeze(-1)  # [batch]
            scores = torch.sigmoid(logits)

        return [float(s) for s in scores.cpu()]

    def score_pairs(
        self,
        queries: list[Item],
        docs: list[Item],
        *,
        instruction: str | None = None,
        options: dict[str, Any] | None = None,
    ) -> ScoreOutput:
        """Score (query, doc) pairs in a batch.

        Batched version for cross-request batching.

        Args:
            queries: Query items.
            docs: Document items.
            instruction: Optional instruction.
            options: Runtime options.

        Returns:
            ScoreOutput with batched scores.
        """
        if self._model is None:
            raise RuntimeError(_ERR_NOT_LOADED)

        pairs = []
        for query, doc in zip(queries, docs, strict=True):
            query_text = self._extract_text(query)
            if instruction:
                query_text = f"{instruction} {query_text}"
            pairs.append((query_text, self._extract_text(doc)))

        inputs = self._tokenize_pairs(pairs)

        with torch.inference_mode():
            outputs = self._model(**inputs)
            logits = outputs.logits.squeeze(-1)
            scores = torch.sigmoid(logits)

        return ScoreOutput(scores=scores.float().cpu().numpy())

    def extract(
        self,
        items: list[Item],
        *,
        labels: list[str] | None = None,
        output_schema: dict[str, Any] | None = None,
        instruction: str | None = None,
        options: dict[str, Any] | None = None,
        prepared_items: list[Any] | None = None,
    ) -> ExtractOutput:
        """Extract pruning spans for (query, document) pairs.

        Uses the instruction parameter as the query. Each document (item) is
        paired with the query, run through the frozen cross-encoder + PruningHead,
        and the results are returned as span-level entities:

        Entity labels:
            "kept"       — text span kept by the pruner (prob >= prune_threshold)
            "highlight"  — high-confidence span (prob >= highlight_threshold)
            "pruned"     — text span removed by the pruner (prob < prune_threshold)
            "summary"    — one per document with rerank score and compression stats

        Each span entity has start/end character offsets into the original text.
        The score field contains the average token probability for that span.

        Options:
            prune_threshold: Override default pruning threshold (0.6)
            highlight_threshold: Override default highlighting threshold (0.9)

        Returns:
            ExtractOutput with Entity spans per item.
        """
        if self._model is None or self._pruning_head is None:
            raise RuntimeError(_ERR_NOT_LOADED)

        opts = options or {}
        query_text = instruction or ""
        prune_threshold = opts.get("prune_threshold", self._prune_threshold)
        highlight_threshold = opts.get("highlight_threshold", self._highlight_threshold)

        # Build pairs: (query, doc) for each item
        doc_texts = [self._extract_text(item) for item in items]
        pairs = [(query_text, doc_text) for doc_text in doc_texts]
        inputs = self._tokenize_pairs(pairs)

        with torch.inference_mode():
            # Forward through base model with hidden states
            outputs = self._model(**inputs, output_hidden_states=True)

            # 1. Rerank scores from classification head
            logits = outputs.logits.squeeze(-1)
            rerank_scores = torch.sigmoid(logits)

            # 2. Token-level pruning from last hidden state
            last_hidden = outputs.hidden_states[-1]  # [batch, seq_len, hidden_size]
            attention_mask = inputs["attention_mask"]
            token_probs = self._pruning_head(last_hidden, attention_mask)

        from sie_server.types.responses import Entity

        all_entities: list[list[Entity]] = []

        for i in range(len(items)):
            entities: list[Entity] = []
            doc_text = doc_texts[i]
            item_probs = token_probs[i].cpu().numpy()
            item_mask = attention_mask[i].cpu().numpy().astype(bool)
            input_ids = inputs["input_ids"][i].cpu().tolist()

            # Find where the passage tokens start (skip [BOS] query [SEP])
            passage_start_idx = self._find_passage_start(
                inputs["input_ids"][i]
            )

            # Get passage token IDs, probs, and decode to find character offsets
            passage_ids = input_ids[passage_start_idx:]
            passage_probs = item_probs[passage_start_idx:]
            passage_mask = item_mask[passage_start_idx:]

            # Decode tokens to spans with character offsets
            spans = self._tokens_to_char_spans(
                passage_ids, passage_probs, passage_mask, doc_text
            )

            # Group consecutive tokens into contiguous spans by label
            kept_count = 0
            total_count = 0
            for span_text, char_start, char_end, avg_prob in spans:
                total_count += 1
                if avg_prob >= highlight_threshold:
                    label = "highlight"
                    kept_count += 1
                elif avg_prob >= prune_threshold:
                    label = "kept"
                    kept_count += 1
                else:
                    label = "pruned"

                entities.append(
                    Entity(
                        text=span_text,
                        label=label,
                        score=float(avg_prob),
                        start=char_start,
                        end=char_end,
                    )
                )

            # Add summary entity (always first)
            compression = 1.0 - (kept_count / max(total_count, 1))
            summary = Entity(
                text=f"rerank={float(rerank_scores[i].cpu()):.4f} "
                     f"compression={compression:.2%} "
                     f"kept={kept_count}/{total_count}",
                label="summary",
                score=float(rerank_scores[i].cpu()),
            )
            entities.insert(0, summary)

            all_entities.append(entities)

        return ExtractOutput(entities=all_entities)

    def _tokens_to_char_spans(
        self,
        token_ids: list[int],
        token_probs: np.ndarray,
        token_mask: np.ndarray,
        original_text: str,
    ) -> list[tuple[str, int, int, float]]:
        """Convert tokens + probabilities to character-level spans.

        Groups consecutive tokens with the same classification (kept/pruned)
        into contiguous spans with character offsets into the original text.

        Returns:
            List of (span_text, char_start, char_end, avg_probability) tuples.
        """
        if not token_ids:
            return []

        # Use tokenizer's offset mapping for precise character alignment
        # Re-tokenize just the passage to get offset_mapping
        # (offset_mapping from pair encoding is unreliable for passage-only offsets)
        passage_encoding = self._tokenizer(
            original_text,
            return_offsets_mapping=True,
            max_length=self._max_seq_length,
            truncation=True,
            add_special_tokens=False,
        )
        offset_mapping = passage_encoding.get("offset_mapping", [])

        # If offset mapping is shorter than our tokens, pad with estimates
        while len(offset_mapping) < len(token_ids):
            offset_mapping.append((0, 0))

        # Group consecutive tokens by classification
        spans: list[tuple[str, int, int, float]] = []
        current_start: int | None = None
        current_end: int | None = None
        current_probs: list[float] = []

        for j, (tok_id, prob, mask) in enumerate(
            zip(token_ids, token_probs, token_mask, strict=False)
        ):
            if not mask:
                continue

            # Skip special tokens (pad, sep, cls/bos)
            if tok_id in (
                self._tokenizer.pad_token_id,
                self._tokenizer.sep_token_id,
                self._tokenizer.cls_token_id,
                self._tokenizer.bos_token_id,
            ):
                continue

            if j < len(offset_mapping):
                char_start, char_end = offset_mapping[j]
            else:
                continue  # Skip tokens beyond offset mapping

            if char_start == char_end == 0:
                continue  # Skip unmappable tokens

            if current_start is None:
                # Start new span
                current_start = char_start
                current_end = char_end
                current_probs = [float(prob)]
            elif char_start <= current_end + 1:
                # Extend current span (consecutive or overlapping)
                current_end = max(current_end, char_end)
                current_probs.append(float(prob))
            else:
                # Gap — emit current span and start new one
                avg_prob = sum(current_probs) / len(current_probs)
                span_text = original_text[current_start:current_end]
                spans.append((span_text, current_start, current_end, avg_prob))

                current_start = char_start
                current_end = char_end
                current_probs = [float(prob)]

        # Emit final span
        if current_start is not None and current_probs:
            avg_prob = sum(current_probs) / len(current_probs)
            span_text = original_text[current_start:current_end]
            spans.append((span_text, current_start, current_end, avg_prob))

        return spans

    def _tokenize_pairs(self, pairs: list[tuple[str, str]]) -> dict[str, torch.Tensor]:
        """Tokenize (query, doc) pairs for the cross-encoder."""
        queries = [p[0] for p in pairs]
        docs = [p[1] for p in pairs]

        inputs = self._tokenizer(
            queries,
            docs,
            max_length=self._max_seq_length,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        return {k: v.to(self._device) for k, v in inputs.items()}

    def _find_passage_start(self, input_ids: torch.Tensor) -> int:
        """Find the start index of passage tokens (after query + separators).

        For XLM-RoBERTa tokenizer (used by BGE-reranker-v2-m3):
        - BOS token (0) at position 0
        - SEP token (2) separates query from passage
        """
        ids = input_ids.cpu().tolist()
        sep_token_id = self._tokenizer.sep_token_id or 2

        # Find the second separator (after query)
        sep_count = 0
        for idx, token_id in enumerate(ids):
            if token_id == sep_token_id:
                sep_count += 1
                if sep_count == 2:  # After query SEP
                    return idx + 1

        # Fallback: skip first 10% of tokens (rough heuristic)
        return max(1, len(ids) // 10)

    def _extract_text(self, item: Item) -> str:
        """Extract text from an Item."""
        text = item.get("text")
        if text is None:
            raise ValueError(_ERR_REQUIRES_TEXT)
        return text

    def get_preprocessor(self) -> CharCountPreprocessor:
        """Return preprocessor for cost estimation."""
        return CharCountPreprocessor(model_name=self._model_name_or_path)
