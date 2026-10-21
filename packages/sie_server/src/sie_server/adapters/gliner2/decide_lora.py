"""Opt-in declared runtime LoRA for the native GLiNER2.5 decision path."""

from __future__ import annotations

import importlib.metadata
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from transformers import BertModel, DebertaV2Model, ModernBertModel

from sie_server.adapters._peft_lora_runtime import DetachedLoraRuntime
from sie_server.adapters.decision_lora.adapter import DecisionLoraAdapter
from sie_server.adapters.gliner2.decide import GLiNER2DecideAdapter
from sie_server.adapters.gliner2.decisions import parse_request
from sie_server.types.inputs import InvalidInputError, Item


class GLiNER2DecideLora(DetachedLoraRuntime):
    """Native encoder, span and classifier Linear deltas; no replaced heads."""

    def _model_targets(self, model: Any) -> tuple[str, ...]:
        if importlib.metadata.version("gliner2") != "2.0.0":
            raise ValueError("Decision LoRA requires GLiNER2 2.0.0")
        from gliner2 import BoundaryExtractor, SpanExtractor

        if type(model) not in (SpanExtractor, BoundaryExtractor):
            raise ValueError("Decision LoRA requires a native GLiNER2 span or boundary extractor")
        heads = ("classifier.", "span_rep.") if type(model) is SpanExtractor else ("classifier.",)
        if type(model.encoder) is ModernBertModel:
            return ("encoder.layers.", *heads)
        if type(model.encoder) in (BertModel, DebertaV2Model):
            return ("encoder.encoder.layer.", *heads)
        raise ValueError("Decision LoRA supports ordinary BERT, DeBERTaV2 and ModernBERT encoders")


class GLiNER2DecideLoraAdapter(DecisionLoraAdapter, GLiNER2DecideAdapter):
    """Keep native decision processing and the ordinary profile's separate execution path."""

    def __init__(
        self,
        model_name_or_path: str | Path,
        *,
        lora_paths: dict[str, str],
        lora_revisions: dict[str, str | None] | None = None,
        max_loaded_loras: int = 10,
        **kwargs: Any,
    ) -> None:
        allowed = {
            "max_seq_length",
            "max_prompt_tokens",
            "inference_batch_tokens",
            "compute_precision",
            "revision",
            "cuda_graphs",
        }
        if kwargs.keys() - allowed:
            raise ValueError(f"Decision LoRA does not accept load options {sorted(kwargs.keys() - allowed)!r}")
        self._lora = GLiNER2DecideLora(lora_paths, lora_revisions, max_loaded_loras)
        super().__init__(model_name_or_path, **kwargs)

    def _lora_model(self) -> Any:
        return self._model

    def _validate_decision_request(
        self,
        items: list[Item],
        *,
        labels: list[str] | None,
        output_schema: dict[str, Any] | None,
        instruction: str | None,
        options: dict[str, Any],
    ) -> None:
        allowed = {
            "label_groups",
            "classification_type",
            "classification_task",
            "multi_label",
            "threshold",
            "group_encoding",
            "examples",
            "overflow_policy",
        }
        if set(options) - allowed:
            raise InvalidInputError("Unsupported GLiNER2 decision option")
        parse_request(labels=labels, output_schema=output_schema, instruction=instruction, options=options)
        policy = options.get("overflow_policy", "truncate_text")
        if not isinstance(policy, str) or policy not in {"truncate_text", "error"}:
            raise InvalidInputError("Decision overflow_policy must be truncate_text or error")
        if not items:
            raise InvalidInputError("Decision LoRA requires at least one item")

    def prime_extract_groups(self, groups: Sequence[Mapping[str, Any]]) -> None:
        """Keep base-only fusion; never prime logits across different adapter selections.

        The native cache key contains token IDs and label positions, but no LoRA
        identity. A mixed batch therefore uses its normal per-selection forwards.
        """
        with self._lora.lock:
            if self._primed is not None:
                self._primed.clear()
            if any(self._request_selection(group.get("options"))[0] is not None for group in groups):
                return
            self._lora.select(None)
            super().prime_extract_groups(groups)
