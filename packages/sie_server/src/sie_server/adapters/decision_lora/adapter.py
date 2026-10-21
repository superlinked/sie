"""Lifecycle and selection for explicitly configured native decision LoRA profiles."""

from __future__ import annotations

from abc import abstractmethod
from typing import Any

from sie_server.adapters._base_adapter import BaseAdapter
from sie_server.adapters._peft_lora_runtime import DetachedLoraRuntime
from sie_server.adapters.lora import LoraRuntimeConfig
from sie_server.core.inference_output import ExtractOutput
from sie_server.types.inputs import InvalidInputError, Item
from sie_server.types.lora import InvalidLoraSelectionError, split_lora_selection


class DecisionLoraAdapter(BaseAdapter):
    """Cooperative base for opt-in profiles; native adapter classes never import it.

    Each concrete class precedes its ordinary adapter in the MRO. The raw model
    still owns inference; the separate controller owns reversible LoRA layers.
    Base CUDA graphs are retained, but an adapted forward always runs eagerly.
    """

    _lora: DetachedLoraRuntime

    @abstractmethod
    def _lora_model(self) -> Any:
        """Return the exact native serving owner after load."""

    @abstractmethod
    def _validate_decision_request(
        self,
        items: list[Item],
        *,
        labels: list[str] | None,
        output_schema: dict[str, Any] | None,
        instruction: str | None,
        options: dict[str, Any],
    ) -> None:
        """Check native task intent before a missing adapter can be fetched."""

    def _request_selection(self, options: dict[str, Any] | None) -> tuple[str | None, dict[str, Any]]:
        if options is not None and not isinstance(options, dict):
            raise InvalidInputError("Decision options must be an object")
        try:
            selection, opts = split_lora_selection(options or {})
        except InvalidLoraSelectionError as error:
            raise InvalidInputError(str(error)) from error
        selected = selection.value
        if selected is not None:
            try:
                source = self._lora.source(selected)
                if "lora_id" in selection.fields and source != selected:
                    raise ValueError("Legacy lora_id must name the declared source, not a served alias")
            except ValueError as error:
                raise InvalidInputError(str(error)) from error
        return selected, opts

    def validate_extract_request(
        self,
        items: list[Item],
        *,
        labels: list[str] | None = None,
        output_schema: dict[str, Any] | None = None,
        instruction: str | None = None,
        options: dict[str, Any] | None = None,
    ) -> None:
        _, opts = self._request_selection(options)
        self._validate_decision_request(
            items, labels=labels, output_schema=output_schema, instruction=instruction, options=opts
        )

    def load(self, device: str) -> None:
        try:
            super().load(device)
            self._lora.bind(self._lora_model())
        except BaseException:
            self.unload()
            raise

    def unload(self) -> None:
        try:
            self._lora.close()
        finally:
            super().unload()

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
        selected, opts = self._request_selection(options)
        self._validate_decision_request(
            items, labels=labels, output_schema=output_schema, instruction=instruction, options=opts
        )
        with self._lora.lock:
            self._lora.select(selected)
            graphs = getattr(self, "_graphs", None)
            if selected is not None:
                if graphs is not None:
                    self._graphs = None
                primed = getattr(self, "_primed", None)
                if primed is not None:
                    primed.clear()
            try:
                return super().extract(
                    items,
                    labels=labels,
                    output_schema=output_schema,
                    instruction=instruction,
                    options=opts,
                    prepared_items=prepared_items,
                )
            finally:
                try:
                    self._lora.select(None)
                finally:
                    if graphs is not None:
                        self._graphs = graphs
                    if selected is not None:
                        primed = getattr(self, "_primed", None)
                        if primed is not None:
                            primed.clear()

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
