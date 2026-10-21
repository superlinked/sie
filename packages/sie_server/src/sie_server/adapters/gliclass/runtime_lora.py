"""Opt-in declared runtime LoRA for native GLiClass uni-encoder decisions."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from transformers import BertModel, DebertaV2Model, ModernBertModel

from sie_server.adapters._peft_lora_runtime import DetachedLoraRuntime
from sie_server.adapters.decision_lora.adapter import DecisionLoraAdapter
from sie_server.adapters.gliclass import GLiClassAdapter
from sie_server.types.inputs import InvalidInputError, Item


class GLiClassLora(DetachedLoraRuntime):
    """Adapt only ordinary encoder projections, retaining the native scoring head."""

    def _model_targets(self, model: Any) -> tuple[str, ...]:
        from gliclass.model import GLiClassModel, GLiClassUniEncoder

        if type(model) is not GLiClassModel or type(model.model) is not GLiClassUniEncoder:
            raise ValueError("GLiClass LoRA requires the native uni-encoder model")
        encoder = model.model.encoder_model
        if type(encoder) is ModernBertModel:
            return ("model.encoder_model.layers.",)
        if type(encoder) in (BertModel, DebertaV2Model):
            return ("model.encoder_model.encoder.layer.",)
        raise ValueError("GLiClass LoRA supports ordinary BERT, DeBERTaV2 and ModernBERT encoders")


class GLiClassLoraAdapter(DecisionLoraAdapter, GLiClassAdapter):
    """Native labels and grouped decisions with explicit runtime LoRA declarations."""

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
            "classification_type",
            "threshold",
            "max_seq_length",
            "compute_precision",
            "revision",
            "cuda_graphs",
            "modernbert_flash",
        }
        if kwargs.keys() - allowed:
            raise ValueError(f"GLiClass LoRA does not accept load options {sorted(kwargs.keys() - allowed)!r}")
        self._lora = GLiClassLora(lora_paths, lora_revisions, max_loaded_loras)
        super().__init__(model_name_or_path, **kwargs)

    def _lora_model(self) -> Any:
        return self._require_pipe().model

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
            "group_encoding",
            "classification_type",
            "threshold",
            "examples",
            "cuda_graphs",
            "overflow_policy",
        }
        if set(options) - allowed:
            raise InvalidInputError("Unsupported GLiClass decision option")
        if output_schema is not None:
            raise InvalidInputError("GLiClass uses labels or options.label_groups, not output_schema")
        groups = self._validate_label_groups(options.get("label_groups"))
        encoding = self._validate_group_encoding(options, groups)
        classification_type = self._validate_classification_type(
            options.get("classification_type", self._classification_type)
        )
        if groups is None:
            normalized_labels = self._validate_labels(labels)
        elif labels:
            raise InvalidInputError("GLiClass accepts either labels or options.label_groups, not both")
        else:
            normalized_labels = self._flatten_label_groups(groups, classification_type)
        if groups is not None and encoding == "separate":
            self._check_group_count(groups)
            for _, group_labels in groups:
                self._check_label_size(group_labels)
        else:
            self._check_label_size(normalized_labels)
        prompt = self._validate_instruction(instruction)
        examples = self._validate_examples(options.get("examples"), normalized_labels, groups)
        self._check_context_size(prompt, examples)
        self._validate_threshold(options.get("threshold", self._threshold))
        self._request_cuda_graphs(options)
        policy = options.get("overflow_policy", "default")
        if not isinstance(policy, str) or policy not in {"default", "truncate_text", "error"}:
            raise InvalidInputError("GLiClass overflow_policy must be default, truncate_text or error")
        if not items:
            raise InvalidInputError("GLiClass LoRA requires at least one item")
        for item in items:
            self._extract_text(item)
