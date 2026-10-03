from typing import Any

import msgspec

from sie_server.core.extract_cost import MAX_EXTRACT_LABELS, output_schema_shape_error
from sie_server.core.score_cost import MAX_SCORE_ITEMS
from sie_server.types.inputs import Item, item_size_error

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
        _check_item_sizes(self.items)


# -- Score --------------------------------------------------------------------


class ScoreRequest(msgspec.Struct):
    query: Item
    items: list[Item]
    instruction: str | None = None
    options: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        if not self.items:
            raise msgspec.ValidationError("Field 'items' must not be empty")
        if self.options is not None and "instruction" in self.options:
            try:
                msgspec.convert(self.options["instruction"], type=str | None, strict=True)
            except msgspec.ValidationError as exc:
                raise msgspec.ValidationError(f"{exc} - at `$.options.instruction`") from exc
        if len(self.items) > MAX_SCORE_ITEMS:
            raise msgspec.ValidationError(f"Field 'items' must contain at most {MAX_SCORE_ITEMS} candidates")
        if error := item_size_error(self.query, "query"):
            raise msgspec.ValidationError(error)
        _check_item_sizes(self.items)


# -- Extract ------------------------------------------------------------------


class ExtractParams(msgspec.Struct):
    labels: list[str] | None = None
    output_schema: dict[str, Any] | None = None
    instruction: str | None = None
    options: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        if self.labels is not None and len(self.labels) > MAX_EXTRACT_LABELS:
            raise msgspec.ValidationError(f"Field 'labels' must contain at most {MAX_EXTRACT_LABELS} labels")
        if self.options is not None and "instruction" in self.options:
            try:
                msgspec.convert(self.options["instruction"], type=str | None, strict=True)
            except msgspec.ValidationError as exc:
                raise msgspec.ValidationError(f"{exc} - at `$.params.options.instruction`") from exc
        if self.output_schema is not None and (error := output_schema_shape_error(self.output_schema)):
            raise msgspec.ValidationError(f"Field {error}")


class ExtractRequest(msgspec.Struct):
    items: list[Item]
    params: ExtractParams | None = None

    def __post_init__(self) -> None:
        if not self.items:
            raise msgspec.ValidationError("Field 'items' must not be empty")
        _check_item_sizes(self.items)


def _check_item_sizes(items: list[Item]) -> None:
    """Reject the request when any item carries more text than the per-item bound.

    Runs as the body is decoded, before anything tokenizes an item, estimates
    its cost, or batches it. One oversized item fails the whole request, as
    the gateway does when a queue work item fails with ``INVALID_INPUT``.
    """
    for index, item in enumerate(items):
        if error := item_size_error(item, f"items[{index}]"):
            raise msgspec.ValidationError(error)
