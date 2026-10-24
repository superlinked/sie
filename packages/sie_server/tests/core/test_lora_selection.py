from itertools import combinations

import pytest
from sie_server.types.lora import (
    LORA_SELECTION_FIELDS,
    InvalidLoraSelectionError,
    split_lora_selection,
)


@pytest.mark.parametrize("field", LORA_SELECTION_FIELDS)
@pytest.mark.parametrize("value", [None, "legal", "org/pinned-adapter"])
def test_selection_preserves_presence_value_and_caller_options(field: str, value: str | None) -> None:
    nested = {"schema": ["unchanged"]}
    options = {field: value, "other": nested}
    selection, remaining = split_lora_selection(options)

    assert selection.provided
    assert selection.value == value
    assert selection.fields == (field,)
    assert remaining == {"other": nested}
    assert remaining["other"] is nested
    assert options == {field: value, "other": nested}
    remaining["new"] = True
    assert "new" not in options


def test_absent_selection_is_distinct_from_explicit_null() -> None:
    missing, missing_options = split_lora_selection({"threshold": 0.5})
    base, base_options = split_lora_selection({"lora_adapter": None, "threshold": 0.5})
    assert not missing.provided
    assert missing.fields == ()
    assert missing.value is None
    assert base.provided
    assert base.value is None
    assert missing_options == base_options == {"threshold": 0.5}


@pytest.mark.parametrize("fields", [*combinations(LORA_SELECTION_FIELDS, 2), LORA_SELECTION_FIELDS])
@pytest.mark.parametrize("value", [None, "legal"])
def test_identical_aliases_are_accepted(fields: tuple[str, ...], value: str | None) -> None:
    selected, remaining = split_lora_selection(dict.fromkeys(fields, value))
    assert selected.value == value
    assert selected.fields == fields
    assert selected.provided
    assert remaining == {}


@pytest.mark.parametrize("fields", list(combinations(LORA_SELECTION_FIELDS, 2)))
@pytest.mark.parametrize("values", [(None, "legal"), ("legal", None), ("legal", "other")])
def test_conflicting_aliases_are_rejected(fields: tuple[str, ...], values: tuple[str | None, ...]) -> None:
    with pytest.raises(InvalidLoraSelectionError, match="must agree"):
        split_lora_selection(dict(zip(fields, values, strict=True)))


@pytest.mark.parametrize("field", LORA_SELECTION_FIELDS)
@pytest.mark.parametrize("value", ["", " \t", 0, False, True, 1.0, [], {}, ["legal"]])
def test_invalid_selector_types_are_rejected(field: str, value: object) -> None:
    with pytest.raises(InvalidLoraSelectionError, match="non-empty string or null"):
        split_lora_selection({field: value})
