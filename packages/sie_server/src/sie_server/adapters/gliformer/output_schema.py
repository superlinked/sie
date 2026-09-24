"""JSON Schema support for GLiFormer structured extraction.

GLiFormer answers an ``output_schema`` with two of its task heads in the same
forward pass:

* span-valued properties (strings, arrays of strings, nested objects, and
  arrays of objects) become a GLiFormer structuring template rooted at
  ``$root``, which the structuring head decodes into one object per document;
* root-level string ``enum`` properties become named classification groups,
  so each one is answered with exactly one of its choices.

This module compiles the supported JSON Schema subset into that request and
shapes the decoded values back into a schema-valid object. It has no runtime
dependency on the ``gliformer`` package so the contract is testable without
model weights.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Final, Literal

from sie_server.core.extract_cost import MAX_EXTRACT_LABELS, output_schema_shape_error
from sie_server.types.inputs import InvalidInputError

FieldKind = Literal["string", "string_array", "choice", "object", "object_array"]

_ANNOTATION_KEYWORDS: Final[frozenset[str]] = frozenset({"title", "description", "default", "examples"})
_OBJECT_KEYWORDS: Final[frozenset[str]] = _ANNOTATION_KEYWORDS | {
    "type",
    "properties",
    "required",
    "additionalProperties",
}
_ROOT_KEYWORDS: Final[frozenset[str]] = _OBJECT_KEYWORDS | {"$defs", "definitions", "$schema", "$id"}
_STRING_KEYWORDS: Final[frozenset[str]] = _ANNOTATION_KEYWORDS | {"type", "enum"}
_ARRAY_KEYWORDS: Final[frozenset[str]] = _ANNOTATION_KEYWORDS | {"type", "items"}
_REF_PREFIXES: Final[tuple[str, ...]] = ("#/$defs/", "#/definitions/")

# GLiFormer reads a template object whose keys all come from its legacy
# descriptor vocabulary as a descriptor rather than as field names.
_DESCRIPTOR_KEYS: Final[frozenset[str]] = frozenset({"description", "children", "fields", "required_fields"})
_DESCRIPTOR_MARKERS: Final[frozenset[str]] = frozenset({"children", "fields", "required_fields"})

# Bounds that keep compiling a caller's schema cheap. Shared ``$ref`` targets
# are expanded as a tree, so properties and expansions are counted while
# compiling, not afterwards.
MAX_SCHEMA_DEPTH: Final[int] = 32
MAX_REF_EXPANSIONS: Final[int] = MAX_EXTRACT_LABELS
# Longest label, field name, or choice. These become prompt text, which is not
# billed, so every one of them is bounded.
MAX_LABEL_CHARS: Final[int] = 128
# Longest caller-supplied text echoed back in an error message.
_ECHO_CHARS: Final[int] = 120

# Key under which a root object template is passed to GLiFormer. The package
# decodes it into a single object per document rather than a list of records.
ROOT_TEMPLATE_KEY: Final[str] = "$root"
_TEMPLATE_STRING: Final[str] = "str"


@dataclass(frozen=True)
class SchemaField:
    """One compiled ``output_schema`` node."""

    kind: FieldKind
    properties: tuple[tuple[str, SchemaField], ...] = ()
    required: frozenset[str] = frozenset()
    choices: tuple[str, ...] = ()


@dataclass(frozen=True)
class StructuredPlan:
    """GLiFormer inputs for one ``output_schema``.

    Attributes:
        root: The compiled root object.
        structures: ``{"$root": template}`` for the structuring head, or
            ``None`` when every property is answered by classification.
        choice_groups: Root ``enum`` properties as ``{name: choices}``
            classification groups, in schema order.
    """

    root: SchemaField
    structures: dict[str, Any] | None
    choice_groups: dict[str, list[str]] = field(default_factory=dict)


def compile_output_schema(output_schema: Any) -> StructuredPlan:
    """Validate ``output_schema`` and build the GLiFormer request for it.

    Raises:
        InvalidInputError: The schema is outside the supported subset.
    """
    if not isinstance(output_schema, Mapping):
        raise InvalidInputError("GLiFormer output_schema must be a JSON object")
    if shape_error := output_schema_shape_error(output_schema):
        raise InvalidInputError(f"GLiFormer {shape_error}")
    root = _SchemaCompiler(output_schema).compile_root()

    template = _template(root)
    return StructuredPlan(
        root=root,
        structures={ROOT_TEMPLATE_KEY: template} if template else None,
        choice_groups={name: list(child.choices) for name, child in root.properties if child.kind == "choice"},
    )


def shape_structured_output(
    plan: StructuredPlan,
    structured: Any,
    choices: Mapping[str, str],
) -> tuple[dict[str, Any], list[str]]:
    """Build the schema-valid ``data`` object for one document.

    A property is present only when the model extracted a value for it.
    Nested objects and array records that lack one of their ``required``
    properties are discarded, as GLiFormer itself does for record arrays.

    Args:
        plan: The compiled schema.
        structured: GLiFormer's decoded ``$root`` object (``{}`` when nothing
            was found, ``None`` when no structuring was requested).
        choices: The selected label for each answered root ``enum`` property.

    Returns:
        ``(data, missing_required)`` where ``missing_required`` lists the root
        ``required`` properties that were not extracted.

    Raises:
        RuntimeError: GLiFormer returned values of the wrong shape.
    """
    raw = {} if structured is None else structured
    if not isinstance(raw, Mapping):
        raise RuntimeError("GLiFormer returned malformed structured output")

    data: dict[str, Any] = {}
    for name, child in plan.root.properties:
        if child.kind == "choice":
            label = choices.get(name)
            if label is not None:
                data[name] = label
            continue
        value = _shape_value(child, raw.get(name))
        if value is not None:
            data[name] = value
    missing = [name for name, _ in plan.root.properties if name in plan.root.required and name not in data]
    return data, missing


class _SchemaCompiler:
    def __init__(self, root: Mapping[str, Any]) -> None:
        self._root = root
        # ``$ref`` targets currently being compiled, to reject recursive schemas.
        self._active_refs: list[str] = []
        self.property_count = 0
        self.ref_expansions = 0

    def compile_root(self) -> SchemaField:
        refs: list[str] = []
        root = self._resolve(self._root, "output_schema", refs)
        if root.get("type") != "object":
            raise InvalidInputError("GLiFormer output_schema root type must be object")
        with self._tracking(refs):
            return self._compile_object(root, "output_schema", depth=1, allowed=_ROOT_KEYWORDS, root=True)

    @contextmanager
    def _tracking(self, refs: list[str]) -> Iterator[None]:
        self._active_refs.extend(refs)
        try:
            yield
        finally:
            del self._active_refs[len(self._active_refs) - len(refs) :]

    def _resolve(self, node: Any, path: str, refs: list[str]) -> Mapping[str, Any]:
        """Inline a local ``$ref`` (as emitted by Pydantic) and drop ``null`` unions.

        Every followed reference is appended to ``refs`` so the caller can
        keep it active while compiling the resolved subtree.
        """
        if not isinstance(node, Mapping):
            raise InvalidInputError(f"GLiFormer {clip(path)} must be an object")
        if "$ref" in node:
            ref = node["$ref"]
            unexpected = set(node) - {"$ref"} - _ANNOTATION_KEYWORDS
            if unexpected:
                raise InvalidInputError(f"GLiFormer {clip(path)} combines $ref with {clip(str(sorted(unexpected)))}")
            if ref in self._active_refs or ref in refs:
                raise InvalidInputError(f"GLiFormer {clip(path)} is recursive, which is not supported")
            self.ref_expansions += 1
            if self.ref_expansions > MAX_REF_EXPANSIONS:
                raise InvalidInputError(f"GLiFormer output_schema expands more than {MAX_REF_EXPANSIONS} $ref values")
            if len(refs) >= MAX_SCHEMA_DEPTH:
                raise InvalidInputError(f"GLiFormer output_schema chains more than {MAX_SCHEMA_DEPTH} $ref values")
            target = self._lookup_ref(ref, path)
            refs.append(ref)
            resolved = self._resolve(target, path, refs)
            return {**resolved, **{key: node[key] for key in _ANNOTATION_KEYWORDS if key in node}}
        if "anyOf" in node:
            options = node["anyOf"]
            if not isinstance(options, list):
                raise InvalidInputError(f"GLiFormer {clip(path)} anyOf must be a list")
            non_null = [option for option in options if option != {"type": "null"}]
            unexpected = set(node) - {"anyOf"} - _ANNOTATION_KEYWORDS
            if len(non_null) != 1 or len(options) != 2 or unexpected:
                raise InvalidInputError(f"GLiFormer {clip(path)} supports anyOf only as an optional (nullable) type")
            resolved = self._resolve(non_null[0], path, refs)
            return {**resolved, **{key: node[key] for key in _ANNOTATION_KEYWORDS if key in node}}
        node_type = node.get("type")
        if isinstance(node_type, list):
            non_null_types = [value for value in node_type if value != "null"]
            if len(non_null_types) != 1 or len(node_type) != 2:
                raise InvalidInputError(f"GLiFormer {clip(path)} supports type lists only as an optional type")
            return {**node, "type": non_null_types[0]}
        return node

    def _lookup_ref(self, ref: Any, path: str) -> Any:
        if not isinstance(ref, str):
            raise InvalidInputError(f"GLiFormer {clip(path)} $ref must be a string")
        for prefix in _REF_PREFIXES:
            if ref.startswith(prefix):
                container = self._root.get(prefix[2:-1])
                name = ref[len(prefix) :]
                if isinstance(container, Mapping) and name in container:
                    return container[name]
                break
        raise InvalidInputError(f"GLiFormer {clip(path)} has an unresolvable $ref {clip(repr(ref))}")

    def _compile_object(
        self,
        node: Mapping[str, Any],
        path: str,
        *,
        depth: int,
        allowed: frozenset[str] = _OBJECT_KEYWORDS,
        root: bool = False,
    ) -> SchemaField:
        if depth > MAX_SCHEMA_DEPTH:
            raise InvalidInputError(f"GLiFormer output_schema nests objects more than {MAX_SCHEMA_DEPTH} levels deep")
        _reject_keywords(node, allowed, path)
        properties = node.get("properties")
        if not isinstance(properties, Mapping) or not properties:
            raise InvalidInputError(f"GLiFormer {clip(path)} requires non-empty properties")
        additional = node.get("additionalProperties", True)
        if not isinstance(additional, bool):
            raise InvalidInputError(f"GLiFormer {clip(path)} additionalProperties must be boolean")
        required: Any = node.get("required", [])
        if (
            not isinstance(required, list)
            or any(not isinstance(name, str) or name not in properties for name in required)
            or len(set(required)) != len(required)
        ):
            raise InvalidInputError(f"GLiFormer {clip(path)} required must list unique property names")
        required_names = frozenset(str(name) for name in required)

        names = set(properties)
        if names <= _DESCRIPTOR_KEYS and names & _DESCRIPTOR_MARKERS:
            raise InvalidInputError(
                f"GLiFormer {clip(path)} cannot consist only of properties named "
                f"{sorted(_DESCRIPTOR_KEYS)}; add another property or rename one"
            )

        compiled: list[tuple[str, SchemaField]] = []
        for name, child in properties.items():
            _validate_property_name(name, path)
            self.property_count += 1
            if self.property_count > MAX_EXTRACT_LABELS:
                raise InvalidInputError(f"GLiFormer output_schema declares more than {MAX_EXTRACT_LABELS} properties")
            child_path = f"{path}.{name}"
            refs: list[str] = []
            resolved = self._resolve(child, child_path, refs)
            with self._tracking(refs):
                compiled.append((name, self._compile_property(resolved, child_path, depth=depth, root=root)))
        return SchemaField(kind="object", properties=tuple(compiled), required=required_names)

    def _compile_property(self, node: Mapping[str, Any], path: str, *, depth: int, root: bool) -> SchemaField:
        node_type = node.get("type")
        if "enum" in node:
            _reject_keywords(node, _STRING_KEYWORDS, path)
            if node_type not in (None, "string"):
                raise InvalidInputError(f"GLiFormer {clip(path)} enum must hold strings")
            if not root:
                raise InvalidInputError(
                    f"GLiFormer {clip(path)}: enum properties are supported only at the schema root, "
                    "where they are answered by document-level classification"
                )
            return SchemaField(kind="choice", choices=_validate_choices(node["enum"], path))
        if node_type == "string":
            _reject_keywords(node, _STRING_KEYWORDS, path)
            return SchemaField(kind="string")
        if node_type == "object":
            return self._compile_object(node, path, depth=depth + 1)
        if node_type == "array":
            _reject_keywords(node, _ARRAY_KEYWORDS, path)
            refs: list[str] = []
            items = self._resolve(node.get("items"), f"{path}[]", refs)
            if "enum" in items:
                raise InvalidInputError(
                    f"GLiFormer {clip(path)}: arrays of enum values are not supported; "
                    "use a single-choice enum property"
                )
            if items.get("type") == "string":
                _reject_keywords(items, _STRING_KEYWORDS, f"{path}[]")
                return SchemaField(kind="string_array")
            if items.get("type") == "object":
                with self._tracking(refs):
                    record = self._compile_object(items, f"{path}[]", depth=depth + 1)
                return SchemaField(kind="object_array", properties=record.properties, required=record.required)
            raise InvalidInputError(f"GLiFormer {clip(path)} arrays must hold strings or objects")
        raise InvalidInputError(
            f"GLiFormer {clip(path)} supports string, string enum, object, and array of string or object values only"
        )


def clip(text: str, limit: int = _ECHO_CHARS) -> str:
    """Shorten caller-supplied text echoed back in an error message."""
    return text if len(text) <= limit else f"{text[:limit]}..."


def _reject_keywords(node: Mapping[str, Any], allowed: frozenset[str], path: str) -> None:
    unsupported = set(node) - allowed
    if unsupported:
        raise InvalidInputError(f"GLiFormer {clip(path)} has unsupported keywords: {clip(str(sorted(unsupported)))}")


def _validate_property_name(name: Any, path: str) -> None:
    # GLiFormer addresses nested fields with dotted paths, so a dot inside a
    # name would be ambiguous.
    if not isinstance(name, str) or not name.strip() or "." in name or len(name) > MAX_LABEL_CHARS:
        raise InvalidInputError(
            f"GLiFormer {clip(path)} property names must be non-empty, contain no '.', "
            f"and have at most {MAX_LABEL_CHARS} characters"
        )


def _validate_choices(values: Any, path: str) -> tuple[str, ...]:
    if (
        not isinstance(values, list)
        or not values
        or any(not isinstance(value, str) or not value.strip() or len(value) > MAX_LABEL_CHARS for value in values)
        or len(set(values)) != len(values)
    ):
        raise InvalidInputError(
            f"GLiFormer {clip(path)} enum must list unique non-empty strings of at most {MAX_LABEL_CHARS} characters"
        )
    return tuple(values)


def _template_key(name: str) -> str:
    # The template language marks required fields with a leading "!" and
    # reserves a leading "$"; doubling the marker keeps the literal name.
    return name[0] + name if name[0] in "!$" else name


def _template(node: SchemaField) -> dict[str, Any]:
    template: dict[str, Any] = {}
    for name, child in node.properties:
        key = _template_key(name)
        if child.kind == "string":
            template[key] = _TEMPLATE_STRING
        elif child.kind == "string_array":
            template[key] = [_TEMPLATE_STRING]
        elif child.kind == "object":
            template[key] = _template(child)
        elif child.kind == "object_array":
            template[key] = [_template(child)]
    return template


def _first_candidate(value: Any) -> Any:
    """Pick the best of several decoded candidates for a single-valued field.

    GLiFormer reports every span it found for a field, best first, and merges
    several predicted root objects into one by listing their values.
    """
    while isinstance(value, list):
        value = next((item for item in value if item is not None), None)
    return value


def _shape_value(node: SchemaField, value: Any) -> Any:
    """Return the schema-valid value, or ``None`` when nothing was extracted."""
    if node.kind in ("string", "object"):
        value = _first_candidate(value)
    if value is None:
        return None
    if node.kind == "string":
        if not isinstance(value, str):
            raise RuntimeError("GLiFormer returned malformed structured output")
        return value if value.strip() else None
    if node.kind == "string_array":
        if not isinstance(value, list) or any(item is not None and not isinstance(item, str) for item in value):
            raise RuntimeError("GLiFormer returned malformed structured output")
        strings = [item for item in value if isinstance(item, str) and item.strip()]
        return strings or None
    if node.kind == "object":
        return _shape_object(node, value)
    if node.kind == "object_array":
        if not isinstance(value, Sequence) or isinstance(value, str):
            raise RuntimeError("GLiFormer returned malformed structured output")
        records = [record for record in (_shape_object(node, item) for item in value) if record is not None]
        return records or None
    raise RuntimeError("GLiFormer returned malformed structured output")


def _shape_object(node: SchemaField, value: Any) -> dict[str, Any] | None:
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise RuntimeError("GLiFormer returned malformed structured output")
    shaped: dict[str, Any] = {}
    for name, child in node.properties:
        child_value = _shape_value(child, value.get(name))
        if child_value is not None:
            shaped[name] = child_value
    if not shaped or any(name not in shaped for name in node.required):
        return None
    return shaped
