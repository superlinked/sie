"""Structural validation of model configs against the worker's schema.

``model_config.schema.json`` is the JSON Schema of the worker's
``sie_server.config.model.ModelConfig``; a test keeps it equal to the worker
model. Only unknown keys and value types are enforced: ``required`` is dropped
because append-only writes carry partial bodies, and cross-field rules stay
with the worker. JSON types apply strictly, so values pydantic's lax mode would
coerce (a quoted number, a boolean in an integer field) are rejected.
"""

from __future__ import annotations

import json
from functools import cache
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator
from jsonschema.exceptions import ValidationError, best_match

SCHEMA_PATH = Path(__file__).with_name("model_config.schema.json")
_NULL_BRANCH = {"type": "null"}
_UNKNOWN_FIELD_MESSAGE = "Unknown field: the worker model config schema does not define it."


# Only these JSON Schema keywords contain schemas. Values under const, enum,
# default, and examples are literal data, even when they contain a required key.
_SCHEMA_MAP_KEYWORDS = {"$defs", "definitions", "properties", "patternProperties", "dependentSchemas"}
_SCHEMA_LIST_KEYWORDS = {"allOf", "anyOf", "oneOf", "prefixItems"}
_SCHEMA_KEYWORDS = {
    "additionalProperties",
    "unevaluatedProperties",
    "propertyNames",
    "items",
    "contains",
    "unevaluatedItems",
    "not",
    "if",
    "then",
    "else",
    "contentSchema",
}


def _without_required(node: Any) -> Any:
    if not isinstance(node, dict):
        return node
    partial = {}
    for key, value in node.items():
        if key == "required" and isinstance(value, list):
            continue
        if key in _SCHEMA_MAP_KEYWORDS and isinstance(value, dict):
            partial[key] = {name: _without_required(schema) for name, schema in value.items()}
        elif key in _SCHEMA_LIST_KEYWORDS and isinstance(value, list):
            partial[key] = [_without_required(schema) for schema in value]
        elif key in _SCHEMA_KEYWORDS:
            partial[key] = _without_required(value)
        else:
            partial[key] = value
    return partial


@cache
def _validator() -> Draft202012Validator:
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    return Draft202012Validator(_without_required(schema))


def _loc(path: Any) -> list[str | int]:
    return [part if isinstance(part, (str, int)) else str(part) for part in path]


def _union_branch_errors(error: ValidationError) -> list[ValidationError]:
    schema = error.schema if isinstance(error.schema, dict) else {}
    branches = schema.get(str(error.validator), [])
    by_branch: dict[int, list[ValidationError]] = {}
    for sub_error in error.context or []:
        index = sub_error.relative_schema_path[0]
        if not isinstance(index, int) or branches[index] == _NULL_BRANCH:
            continue
        by_branch.setdefault(index, []).append(sub_error)
    best = best_match(sub_error for sub_errors in by_branch.values() for sub_error in sub_errors)
    if best is None:
        return []
    return by_branch[int(best.relative_schema_path[0])]


def _details(error: ValidationError) -> list[dict[str, Any]]:
    loc = _loc(error.absolute_path)
    if error.validator in ("anyOf", "oneOf"):
        branch_errors = _union_branch_errors(error)
        if branch_errors:
            return [detail for sub_error in branch_errors for detail in _details(sub_error)]
    if error.validator == "additionalProperties" and isinstance(error.instance, dict):
        schema = error.schema if isinstance(error.schema, dict) else {}
        known = schema.get("properties", {})
        unknown = sorted((key for key in error.instance if key not in known), key=str)
        if unknown:
            return [{"loc": [*loc, str(key)], "message": _UNKNOWN_FIELD_MESSAGE} for key in unknown]
    return [{"loc": loc, "message": error.message}]


def model_config_schema_errors(config: dict[str, Any]) -> list[dict[str, Any]]:
    """Return one ``{"loc", "message"}`` entry per field the worker schema rejects."""
    details = [detail for error in _validator().iter_errors(config) for detail in _details(error)]
    return sorted(details, key=lambda detail: ([str(part) for part in detail["loc"]], detail["message"]))
