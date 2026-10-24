"""Request-only LoRA selector handling; authorization remains server-owned."""

from collections.abc import Mapping
from enum import Enum
from typing import Any

LORA_FIELDS = ("lora_adapter", "lora", "lora_id")


class UnsetLora(Enum):
    OMIT = "omit"


LORA_UNSET = UnsetLora.OMIT


def lora_selection(options: Mapping[str, Any] | None) -> str | None | UnsetLora:
    """Validate aliases without erasing absence or an explicit null selection."""
    selected: str | None | UnsetLora = LORA_UNSET
    for field in LORA_FIELDS:
        if options is None or field not in options:
            continue
        value = options[field]
        if value is not None and (not isinstance(value, str) or not value.strip()):
            raise ValueError(f"'{field}' must be a non-empty string or null")
        if selected is not LORA_UNSET and value != selected:
            raise ValueError("LoRA selector fields must agree, including explicit null values")
        selected = value
    return selected


def merge_runtime_lora_options(
    defaults: dict[str, Any] | None, options: dict[str, Any] | None
) -> dict[str, Any] | None:
    """Per-call selection replaces defaults as one setting, regardless of spelling."""
    lora_selection(defaults)
    selected = lora_selection(options)
    if defaults is None:
        return options
    if options is None:
        return defaults
    inherited = (
        defaults
        if selected is LORA_UNSET
        else {key: value for key, value in defaults.items() if key not in LORA_FIELDS}
    )
    return {**inherited, **options}


def merge_lora_extra_body(body: dict[str, Any], extra: dict[str, Any] | None) -> dict[str, Any]:
    """Keep ordinary extra-body overrides, but never silently overwrite LoRA selection."""
    selected = lora_selection(body)
    other = lora_selection(extra)
    if selected is not LORA_UNSET and other is not LORA_UNSET and selected != other:
        raise ValueError("LoRA selection in extra_body conflicts with the typed request selection")
    return {**body, **(extra or {})}
