"""Explicit entity-only GLiNER2 1.3.2/2.0.0 runtime LoRA profile.

Configure this class explicitly with a nonempty ``lora_paths`` map. Declarations
are deferred, never preloaded. Requests select ``options.lora``; ``lora_id`` and
classification, relations, structured output and instruction modes are refused,
including when selecting base. The ordinary GLiNER2 adapters are unchanged.

An opt-in profile sets ``adapter_path`` to
``sie_server.adapters.gliner2.entity_lora:GLiNER2EntityLoraAdapter`` and declares
``adapter_options.loadtime.lora_paths`` using the existing source/revision
format. Revisions use canonical source keys; the loader may include pins from
other profiles, which are validated but not served here. Plain generic PEFT
encoder/span A/B matrices are supported for ordinary BERT and DeBERTaV2
encoders. Saved heads, merged weights and other PEFT transforms are refused.

The first valid cold request returns ``LORA_LOADING`` while the registry loads
the declaration. Clients may retry after readiness; the extract SDK currently
surfaces that response as a ``ServerError``. Base selection restores the original
projection modules while adapter weights remain resident outside the model
tree. Admission and locking still add work; this is not a latency guarantee.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from sie_server.adapters._prompt_limit import DEFAULT_MAX_SCHEMA_PROMPT_TOKENS, check_label_chars
from sie_server.adapters._types import ComputePrecision
from sie_server.adapters.gliner2.adapter import GLiNER2Adapter
from sie_server.adapters.lora import LoraRuntimeConfig
from sie_server.core.inference_output import ExtractOutput
from sie_server.types.inputs import InvalidInputError, Item


class GLiNER2EntityLoraAdapter(GLiNER2Adapter):
    """Entity-only serving with a separate generic PEFT lifecycle controller."""

    def __init__(
        self,
        model_name_or_path: str | Path,
        *,
        lora_paths: dict[str, str],
        lora_revisions: dict[str, str | None] | None = None,
        max_loaded_loras: int = 10,
        threshold: float = 0.5,
        default_labels: list[str] | None = None,
        max_seq_length: int | None = None,
        max_prompt_tokens: int = DEFAULT_MAX_SCHEMA_PROMPT_TOKENS,
        compute_precision: ComputePrecision = "float16",
        revision: str | None = None,
        **kwargs: Any,
    ) -> None:
        if kwargs:
            raise ValueError(f"Entity-only LoRA does not accept load options {sorted(kwargs)!r}")
        # Optional dependencies are imported only for the explicitly chosen
        # variant. Generic GLiNER2/classification/native adapters never import it.
        from sie_server.adapters.gliner2._lora import GLiNER2Lora

        self._lora = GLiNER2Lora(lora_paths, lora_revisions, max_loaded_loras)
        super().__init__(
            model_name_or_path,
            threshold=threshold,
            default_labels=default_labels,
            max_seq_length=max_seq_length,
            max_prompt_tokens=max_prompt_tokens,
            compute_precision=compute_precision,
            revision=revision,
        )

    def load(self, device: str) -> None:
        try:
            super().load(device)
            self._lora.bind(self._model)
        except BaseException:
            self.unload()
            raise

    def unload(self) -> None:
        try:
            self._lora.close()
        finally:
            super().unload()

    def validate_extract_request(
        self,
        items: list[Item],
        *,
        labels: list[str] | None = None,
        output_schema: dict[str, Any] | None = None,
        instruction: str | None = None,
        options: dict[str, Any] | None = None,
    ) -> None:
        """Validate task/source intent before selection or any load-on-miss."""
        if output_schema is not None or instruction is not None:
            raise InvalidInputError("Entity-only LoRA does not accept output_schema or instruction")
        opts = {} if options is None else options
        if not isinstance(opts, dict):
            raise InvalidInputError("Entity-only LoRA options must be an object")
        if opts.get("lora_id") is not None:
            raise InvalidInputError("Entity-only LoRA selects adapters with options.lora, not lora_id")
        if set(opts) - {"threshold", "lora", "lora_id"}:
            raise InvalidInputError("Entity-only LoRA accepts only threshold and lora runtime options")
        if any(item.metadata is not None and "entities" in item.metadata for item in items):
            raise InvalidInputError("Entity-only LoRA does not accept relation entities")
        if not items or any(item.text is None for item in items):
            raise InvalidInputError("Entity-only LoRA requires text items")
        self._validate_threshold(opts.get("threshold", self._threshold))
        effective_labels = self._validate_labels(labels if labels is not None else self._default_labels)
        check_label_chars("GLiNER2", "labels", effective_labels)
        selected = opts.get("lora")
        if selected is not None:
            if not isinstance(selected, str):
                raise InvalidInputError("options.lora must name a declared adapter or be null")
            try:
                self._lora.source(selected)
            except ValueError as error:
                raise InvalidInputError(str(error)) from error

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
        self.validate_extract_request(
            items,
            labels=labels,
            output_schema=output_schema,
            instruction=instruction,
            options=options,
        )
        with self._lora.lock:
            self._lora.select((options or {}).get("lora"))
            try:
                return super().extract(
                    items,
                    labels=labels,
                    options=options,
                    prepared_items=prepared_items,
                )
            finally:
                self._lora.select(None)

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
