"""Dependency-free parsing of public and legacy request LoRA selectors.

Parsing does not authorize an adapter or resolve its source. Callers must check
the selected profile's declarations before loading or dispatching any work.
"""

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

LORA_SELECTION_FIELDS = ("lora_adapter", "lora", "lora_id")


class InvalidLoraSelectionError(ValueError):
    """A request contains an invalid or inconsistent LoRA selector."""


@dataclass(frozen=True)
class LoraSelection:
    """Preserve absence, explicit base selection, and supplied alias spellings."""

    provided: bool
    value: str | None
    fields: tuple[str, ...]


def split_lora_selection(options: Mapping[str, Any]) -> tuple[LoraSelection, dict[str, Any]]:
    """Return the selection and a copy without any request selector fields.

    ``lora_adapter`` is canonical; ``lora`` and ``lora_id`` are compatibility
    spellings. Multiple supplied fields must contain the same string or all
    contain null. An absent selector leaves a profile default intact; explicit
    null requests the base model. Strings are preserved exactly, never trimmed
    or interpreted as paths. ``fields`` lets admission retain any narrower
    legacy source-ID rule for ``lora_id``.
    """
    fields = tuple(field for field in LORA_SELECTION_FIELDS if field in options)
    value: str | None = None
    for index, field in enumerate(fields):
        candidate = options[field]
        if candidate is not None and (not isinstance(candidate, str) or not candidate.strip()):
            raise InvalidLoraSelectionError(f"'{field}' must be a non-empty string or null")
        if index and candidate != value:
            raise InvalidLoraSelectionError("LoRA selector fields must agree, including explicit null values")
        value = candidate
    return (
        LoraSelection(provided=bool(fields), value=value, fields=fields),
        {key: value for key, value in options.items() if key not in LORA_SELECTION_FIELDS},
    )
