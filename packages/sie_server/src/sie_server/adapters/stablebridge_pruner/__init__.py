from __future__ import annotations

import gc
import logging
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
from huggingface_hub import hf_hub_download
from huggingface_hub.errors import HfHubHTTPError, RepositoryNotFoundError
from torch import nn
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from sie_server.adapters.base import ModelAdapter, ModelCapabilities, ModelDims
from sie_server.core.inference_output import ExtractOutput, ScoreOutput
from sie_server.core.preprocessor import CharCountPreprocessor
from sie_server.types.inputs import Item
from sie_server.types.responses import Entity

logger = logging.getLogger(__name__)

ComputePrecision = Literal["float16", "bfloat16", "float32"]

_ERR_NOT_LOADED = "Model not loaded. Call load() first."
_ERR_REQUIRES_TEXT = "StablebridgePrunerAdapter requires text input"

DEFAULT_PRUNE_THRESHOLD = 0.6
DEFAULT_HIGHLIGHT_THRESHOLD = 0.9


class PruningHead(nn.Module):
    """MLP head predicting per-token keep probabilities."""

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
        logits = self.classifier(hidden_states).squeeze(-1)
        probs = torch.sigmoid(logits)
        if attention_mask is not None:
            probs = probs * attention_mask.to(probs.dtype)
        return probs


class StablebridgePrunerAdapter(ModelAdapter):
    """Frozen ``BAAI/bge-reranker-v2-m3`` + trained ``PruningHead`` MLP."""

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
        _ = kwargs
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

        self._model: Any | None = None
        self._pruning_head: PruningHead | None = None
        self._tokenizer: Any | None = None
        self._device: str | None = None

    @property
    def capabilities(self) -> ModelCapabilities:
        return ModelCapabilities(inputs=["text"], outputs=["score", "json"])

    @property
    def dims(self) -> ModelDims:
        return ModelDims()

    def load(self, device: str) -> None:
        self._device = device

        dtype_map = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }
        dtype = dtype_map.get(self._compute_precision, torch.bfloat16)

        self._tokenizer = AutoTokenizer.from_pretrained(
            self._model_name_or_path,
            trust_remote_code=self._trust_remote_code,
        )

        # ``output_hidden_states`` is requested per-call from ``extract``;
        # leaving it off here keeps ``score`` / ``score_pairs`` cheap.
        self._model = AutoModelForSequenceClassification.from_pretrained(
            self._model_name_or_path,
            torch_dtype=dtype,
            trust_remote_code=self._trust_remote_code,
        )
        self._model.to(device)
        self._model.eval()

        self._pruning_head = PruningHead(
            hidden_size=self._hidden_size,
            intermediate_size=self._intermediate_size,
            dropout=self._dropout,
        )
        self._load_pruning_head_weights(dtype)
        self._pruning_head.to(device)
        self._pruning_head.eval()

    def _load_pruning_head_weights(self, dtype: torch.dtype) -> None:
        checkpoint_path = self._resolve_checkpoint_path()
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=True)

        if not isinstance(ckpt, dict):
            msg = f"Unexpected checkpoint type: {type(ckpt)}"
            raise ValueError(msg)

        if "pruning_head" in ckpt:
            state_dict = ckpt["pruning_head"]
        elif "model_state_dict" in ckpt:
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
            state_dict = ckpt

        assert self._pruning_head is not None
        result = self._pruning_head.load_state_dict(state_dict, strict=False)
        if result.missing_keys:
            msg = (
                f"PruningHead checkpoint is missing keys "
                f"(hidden={self._hidden_size}, intermediate={self._intermediate_size}); "
                f"those layers would run with random weights. "
                f"missing={list(result.missing_keys)} "
                f"unexpected={list(result.unexpected_keys)}"
            )
            raise RuntimeError(msg)
        if result.unexpected_keys:
            logger.warning(
                "PruningHead checkpoint has unexpected keys (ignored): %s",
                list(result.unexpected_keys),
            )
        self._pruning_head.to(dtype)

    def _resolve_checkpoint_path(self) -> str:
        local_path = Path(self._pruning_head_path)
        if local_path.is_file():
            return str(local_path)
        if (local_path / self._pruning_head_file).is_file():
            return str(local_path / self._pruning_head_file)

        try:
            return hf_hub_download(
                repo_id=self._pruning_head_path,
                filename=self._pruning_head_file,
            )
        except (HfHubHTTPError, RepositoryNotFoundError, OSError) as e:
            msg = (
                f"Could not find PruningHead weights at "
                f"'{self._pruning_head_path}' "
                f"(file: '{self._pruning_head_file}'): {e}"
            )
            raise FileNotFoundError(msg) from e

    def unload(self) -> None:
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
        _ = options
        if self._model is None:
            raise RuntimeError(_ERR_NOT_LOADED)

        query_text = self._extract_text(query)
        if instruction:
            query_text = f"{instruction} {query_text}"

        pairs = [(query_text, self._extract_text(item)) for item in items]
        inputs = self._tokenize_pairs(pairs)

        with torch.inference_mode():
            outputs = self._model(**inputs)
            logits = outputs.logits.squeeze(-1)
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
        _ = options
        if self._model is None:
            raise RuntimeError(_ERR_NOT_LOADED)

        pairs: list[tuple[str, str]] = []
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
        """Return per-item ``Entity`` spans (kept / highlight / pruned + summary).

        ``instruction`` is treated as the query. ``options`` may override
        ``prune_threshold`` / ``highlight_threshold``.
        """
        _ = labels, output_schema, prepared_items
        if self._model is None or self._pruning_head is None:
            raise RuntimeError(_ERR_NOT_LOADED)

        opts = options or {}
        query_text = instruction or ""
        prune_threshold = float(opts.get("prune_threshold", self._prune_threshold))
        highlight_threshold = float(opts.get("highlight_threshold", self._highlight_threshold))

        doc_texts = [self._extract_text(item) for item in items]
        pairs = [(query_text, doc_text) for doc_text in doc_texts]
        inputs = self._tokenize_pairs(pairs)

        with torch.inference_mode():
            outputs = self._model(**inputs, output_hidden_states=True)
            logits = outputs.logits.squeeze(-1)
            rerank_scores = torch.sigmoid(logits)

            last_hidden = outputs.hidden_states[-1]
            attention_mask = inputs["attention_mask"]
            token_probs = self._pruning_head(last_hidden, attention_mask)

        all_entities: list[list[Entity]] = []

        for i in range(len(items)):
            entities: list[Entity] = []
            doc_text = doc_texts[i]
            item_probs = token_probs[i].float().cpu().numpy()
            item_mask = attention_mask[i].cpu().numpy().astype(bool)
            input_ids = inputs["input_ids"][i].cpu().tolist()

            passage_start_idx = self._find_passage_start(inputs["input_ids"][i])

            passage_ids = input_ids[passage_start_idx:]
            passage_probs = item_probs[passage_start_idx:]
            passage_mask = item_mask[passage_start_idx:]

            spans = self._tokens_to_char_spans(
                passage_ids,
                passage_probs,
                passage_mask,
                doc_text,
                prune_threshold=prune_threshold,
                highlight_threshold=highlight_threshold,
            )

            kept_count = 0
            total_count = 0
            for span_text, char_start, char_end, avg_prob, label in spans:
                total_count += 1
                if label != "pruned":
                    kept_count += 1

                entities.append(
                    Entity(
                        text=span_text,
                        label=label,
                        score=float(avg_prob),
                        start=char_start,
                        end=char_end,
                    )
                )

            compression = 1.0 - (kept_count / max(total_count, 1))
            rerank_score = float(rerank_scores[i].cpu())
            summary = Entity(
                text=(f"rerank={rerank_score:.4f} compression={compression:.2%} kept={kept_count}/{total_count}"),
                label="summary",
                score=rerank_score,
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
        *,
        prune_threshold: float,
        highlight_threshold: float,
    ) -> list[tuple[str, int, int, float, str]]:
        if not token_ids:
            return []

        assert self._tokenizer is not None
        # Pair-encoding offsets are unreliable for passage-only spans; re-tokenize.
        passage_encoding = self._tokenizer(
            original_text,
            return_offsets_mapping=True,
            max_length=self._max_seq_length,
            truncation=True,
            add_special_tokens=False,
        )
        offset_mapping = list(passage_encoding.get("offset_mapping", []))

        while len(offset_mapping) < len(token_ids):
            offset_mapping.append((0, 0))

        def _label_for(p: float) -> str:
            if p >= highlight_threshold:
                return "highlight"
            if p >= prune_threshold:
                return "kept"
            return "pruned"

        spans: list[tuple[str, int, int, float, str]] = []
        current_start: int | None = None
        current_end: int | None = None
        current_probs: list[float] = []
        current_label: str | None = None

        special_ids = {
            self._tokenizer.pad_token_id,
            self._tokenizer.sep_token_id,
            self._tokenizer.cls_token_id,
            self._tokenizer.bos_token_id,
        }

        for j, (tok_id, prob, mask) in enumerate(zip(token_ids, token_probs, token_mask, strict=False)):
            if not mask:
                continue
            if tok_id in special_ids:
                continue
            if j >= len(offset_mapping):
                continue

            char_start, char_end = offset_mapping[j]
            if char_start == char_end == 0:
                continue

            tok_prob = float(prob)
            tok_label = _label_for(tok_prob)

            if current_start is None:
                current_start = char_start
                current_end = char_end
                current_probs = [tok_prob]
                current_label = tok_label
            elif current_end is not None and char_start <= current_end + 1 and current_label == tok_label:
                current_end = max(current_end, char_end)
                current_probs.append(tok_prob)
            else:
                avg_prob = sum(current_probs) / len(current_probs)
                assert current_end is not None
                assert current_label is not None
                span_text = original_text[current_start:current_end]
                spans.append((span_text, current_start, current_end, avg_prob, current_label))

                current_start = char_start
                current_end = char_end
                current_probs = [tok_prob]
                current_label = tok_label

        if current_start is not None and current_probs and current_end is not None and current_label is not None:
            avg_prob = sum(current_probs) / len(current_probs)
            span_text = original_text[current_start:current_end]
            spans.append((span_text, current_start, current_end, avg_prob, current_label))

        return spans

    def _tokenize_pairs(self, pairs: list[tuple[str, str]]) -> dict[str, torch.Tensor]:
        assert self._tokenizer is not None
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
        assert self._tokenizer is not None
        ids = input_ids.cpu().tolist()
        sep_token_id = self._tokenizer.sep_token_id or 2

        sep_count = 0
        for idx, token_id in enumerate(ids):
            if token_id == sep_token_id:
                sep_count += 1
                if sep_count == 2:
                    return idx + 1

        # Fallback only fires for tokenizers without a second SEP; spans may misalign.
        logger.warning(
            "Could not locate second SEP token for model %s; falling back to a 10%% heuristic.",
            self._model_name_or_path,
        )
        return max(1, len(ids) // 10)

    def _extract_text(self, item: Item) -> str:
        if item.text is None:
            raise ValueError(_ERR_REQUIRES_TEXT)
        return item.text

    def get_preprocessor(self) -> CharCountPreprocessor:
        return CharCountPreprocessor(model_name=self._model_name_or_path)
