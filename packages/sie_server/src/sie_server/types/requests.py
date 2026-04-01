from typing import Any

import msgspec

from sie_server.types.inputs import Item

# -- Encode ------------------------------------------------------------------


class EncodeParams(msgspec.Struct):
    output_types: list[str] | None = None
    output_dtype: str | None = None
    instruction: str | None = None
    options: dict[str, Any] | None = None


class EncodeRequest(msgspec.Struct):
    items: list[Item]
    params: EncodeParams | None = None

    def __post_init__(self) -> None:
        if not self.items:
            raise msgspec.ValidationError("Field 'items' must not be empty")


# -- Score --------------------------------------------------------------------


class ScoreRequest(msgspec.Struct):
    query: Item
    items: list[Item]
    instruction: str | None = None
    options: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        if not self.items:
            raise msgspec.ValidationError("Field 'items' must not be empty")


# -- Extract ------------------------------------------------------------------


class ExtractParams(msgspec.Struct):
    labels: list[str] | None = None
    output_schema: dict[str, Any] | None = None
    instruction: str | None = None
    options: dict[str, Any] | None = None


class ExtractRequest(msgspec.Struct):
    items: list[Item]
    params: ExtractParams | None = None

    def __post_init__(self) -> None:
        if not self.items:
            raise msgspec.ValidationError("Field 'items' must not be empty")
