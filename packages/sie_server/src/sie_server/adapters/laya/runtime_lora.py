"""Opt-in declared runtime LoRA for native ModernBERT Laya decisions."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from transformers import ModernBertModel

from sie_server.adapters._peft_lora_runtime import DetachedLoraRuntime
from sie_server.adapters.decision_lora.adapter import DecisionLoraAdapter
from sie_server.adapters.laya.adapter import LayaAdapter, _threshold_option
from sie_server.adapters.laya.model import LayaDecisionHead, LayaModel
from sie_server.types.inputs import InvalidInputError, Item


class LayaLora(DetachedLoraRuntime):
    """Adapt only the ordinary ModernBERT encoder; preserve the typed head."""

    def _model_targets(self, model: Any) -> tuple[str, ...]:
        if (
            type(model) is not LayaModel
            or type(model.encoder) is not ModernBertModel
            or type(model.head) is not LayaDecisionHead
        ):
            raise ValueError("Laya LoRA requires the native ModernBERT encoder and unchanged decision head")
        return ("encoder.layers.",)


class LayaLoraAdapter(DecisionLoraAdapter, LayaAdapter):
    """The ordinary Laya profile remains separate and unchanged."""

    def __init__(
        self,
        model_name_or_path: str | Path,
        *,
        lora_paths: dict[str, str],
        lora_revisions: dict[str, str | None] | None = None,
        max_loaded_loras: int = 10,
        **kwargs: Any,
    ) -> None:
        allowed = {"max_seq_length", "max_forward_tokens", "compute_precision", "revision", "cuda_graphs", "fused_rope"}
        if kwargs.keys() - allowed:
            raise ValueError(f"Laya LoRA does not accept load options {sorted(kwargs.keys() - allowed)!r}")
        self._lora = LayaLora(lora_paths, lora_revisions, max_loaded_loras)
        super().__init__(model_name_or_path, **kwargs)

    def _lora_model(self) -> LayaModel | None:
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
        if set(options) - {"max_len", "head_max_len", "threshold", "overflow_policy"}:
            raise InvalidInputError("Unsupported Laya decision option")
        questions, caller_labels = self._resolve_questions(labels, output_schema, instruction)
        max_len, _ = self._token_budget(options)
        self._check_item_tokens(len(questions), max_len)
        if caller_labels is not None:
            _threshold_option(options)
        policy = options.get("overflow_policy", "truncate_text")
        if not isinstance(policy, str) or policy not in {"truncate_text", "error"}:
            raise InvalidInputError("Laya overflow_policy must be truncate_text or error")
        if not items:
            raise InvalidInputError("Laya LoRA requires at least one item")
