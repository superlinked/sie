"""Architecture boundary for the explicitly selected GLiNER2 entity LoRA profile."""

from __future__ import annotations

import importlib.metadata
from typing import Any

from gliner2 import GLiNER2
from transformers import BertModel, DebertaV2Model

from sie_server.adapters._peft_lora_runtime import DetachedLoraRuntime


class GLiNER2Lora(DetachedLoraRuntime):
    """The existing entity-only architecture and target policy."""

    def _model_targets(self, model: Any) -> tuple[str, ...]:
        if importlib.metadata.version("gliner2") not in {"1.3.2", "2.0.0"}:
            raise ValueError("The entity LoRA adapter requires GLiNER2 1.3.2 or 2.0.0")
        if type(model) is not GLiNER2 or type(model.encoder) not in (BertModel, DebertaV2Model):
            raise ValueError("Entity LoRA requires the ordinary GLiNER2 BERT or DeBERTaV2 encoder")
        return ("encoder.encoder.layer.", "span_rep.")
