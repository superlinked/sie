"""Opt-in declared runtime LoRA for Perplexity's native Qwen3.5 decision backbone."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from sie_server.adapters._peft_lora_runtime import DetachedLoraRuntime
from sie_server.adapters.decision_lora.adapter import DecisionLoraAdapter
from sie_server.adapters.pplx_decider.adapter import PerplexityDecisionAdapter
from sie_server.adapters.pplx_decider.model import DecisionModel
from sie_server.adapters.pplx_decider.questions import parse_questions
from sie_server.types.inputs import InvalidInputError, Item


class PerplexityDecisionLora(DetachedLoraRuntime):
    """Only text-backbone Linear deltas; readout, pooling and calibration stay native."""

    def _model_targets(self, model: Any) -> tuple[str, ...]:
        from transformers.models.qwen3_5.modeling_qwen3_5 import (  # ty: ignore[unresolved-import]
            Qwen3_5Model,
            Qwen3_5TextModel,
        )

        if (
            type(model) is not DecisionModel
            or type(model.backbone) is not Qwen3_5Model
            or type(model.backbone.language_model) is not Qwen3_5TextModel
        ):
            raise ValueError("Perplexity LoRA requires the native Qwen3.5 decision backbone")
        return ("backbone.language_model.layers.",)


class PerplexityDecisionLoraAdapter(DecisionLoraAdapter, PerplexityDecisionAdapter):
    """Runtime adapters are explicitly declared; ordinary decision profiles are unchanged."""

    def __init__(
        self,
        model_name_or_path: str | Path,
        *,
        lora_paths: dict[str, str],
        lora_revisions: dict[str, str | None] | None = None,
        max_loaded_loras: int = 10,
        **kwargs: Any,
    ) -> None:
        allowed = {"max_seq_length", "max_forward_tokens", "compute_precision", "revision"}
        if kwargs.keys() - allowed:
            raise ValueError(f"Perplexity LoRA does not accept load options {sorted(kwargs.keys() - allowed)!r}")
        self._lora = PerplexityDecisionLora(lora_paths, lora_revisions, max_loaded_loras)
        super().__init__(model_name_or_path, **kwargs)

    def _lora_model(self) -> DecisionModel | None:
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
        if labels is not None or instruction is not None:
            raise InvalidInputError("Use typed output_schema questions with their own instructions and criteria")
        if set(options) - {"overflow_policy"} or options.get("overflow_policy", "error") != "error":
            raise InvalidInputError("Perplexity decisions accept only overflow_policy='error'")
        parse_questions(output_schema)
        if not items:
            raise InvalidInputError("Perplexity LoRA requires at least one item")
