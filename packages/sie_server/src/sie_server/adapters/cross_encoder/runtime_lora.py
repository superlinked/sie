"""Explicit runtime-LoRA adapter; ordinary CrossEncoder profiles stay unchanged."""

from pathlib import Path
from typing import Any

from sie_server.adapters._types import AttnImplementation, ComputePrecision
from sie_server.adapters.cross_encoder import CrossEncoderAdapter
from sie_server.adapters.cross_encoder._lora import CrossEncoderLora
from sie_server.adapters.lora import LoraRuntimeConfig


class CrossEncoderLoraAdapter(CrossEncoderAdapter):
    """Opt-in runtime LoRA for a standard, unquantized BERT or XLM-R cross-encoder.

    Select ``sie_server.adapters.cross_encoder.runtime_lora:CrossEncoderLoraAdapter``
    explicitly and declare ``lora_paths={served_name: source}``. Only ordinary
    A/B matrices targeting encoder Linear modules of the exact
    ``BertForSequenceClassification`` or ``XLMRobertaForSequenceClassification``
    class are accepted. The outer CrossEncoder, classifier, activation and scoring methods are inherited;
    trained replacement heads and custom/causal reranker classes are rejected.
    Native BertFlash/default catalog profiles are separate implementations and
    do not gain LoRA support from this adapter.

    ``options.lora=None`` restores the original base modules while adapters stay
    resident off-tree. Loading, selection and eviction use the worker's existing
    mutation guard. This lifecycle contract does not establish zero overhead or
    retrieval quality. Ordinary ``CrossEncoderAdapter`` profiles do not import
    this module or construct any of its state.
    """

    def __init__(
        self,
        model_name_or_path: str | Path,
        *,
        trust_remote_code: bool = False,
        max_length: int | None = None,
        compute_precision: ComputePrecision | None = None,
        attn_implementation: AttnImplementation = "sdpa",
        revision: str | None = None,
        lora_paths: dict[str, str] | None = None,
        lora_revisions: dict[str, str | None] | None = None,
        max_loaded_loras: int = 10,
        **kwargs: Any,
    ) -> None:
        if not lora_paths:
            raise ValueError("CrossEncoderLoraAdapter requires nonempty lora_paths declarations")
        super().__init__(
            model_name_or_path,
            trust_remote_code=trust_remote_code,
            max_length=max_length,
            compute_precision=compute_precision,
            attn_implementation=attn_implementation,
            revision=revision,
            **kwargs,
        )
        self._lora = CrossEncoderLora(lora_paths, lora_revisions, max_loaded_loras)

    def load(self, device: str) -> None:
        super().load(device)
        assert self._model is not None
        self._lora.bind(self._model)

    def unload(self) -> None:
        try:
            self._lora.close()
        finally:
            super().unload()

    def supports_lora(self) -> bool:
        return True

    def supports_hot_lora_reload(self) -> bool:
        return True

    def lora_runtime_config(self) -> LoraRuntimeConfig:
        return self._lora.runtime_config()

    def fetch_lora(self, lora_path: str, revision: str | None = None) -> str:
        return self._lora.fetch(lora_path, revision)

    def discard_lora_source(self, lora_path: str) -> None:
        self._lora.discard(lora_path)

    def load_lora(self, lora_path: str, revision: str | None = None, *, source: str | None = None) -> int:
        return self._lora.load(lora_path, revision, source=source)

    def unload_lora(self, lora_name: str) -> None:
        self._lora.unload(lora_name)

    def set_active_lora(self, lora_name: str | None) -> None:
        self._lora.select(lora_name)
