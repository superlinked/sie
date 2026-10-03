#!/usr/bin/env python3
"""IPC types parity check across the sidecar boundary.

The IPC wire contract is declared twice:

* ``packages/sie_server_sidecar/src/protocol/ipc_types.rs`` (Rust —
  owned by the worker-sidecar; ``src/ipc_types.rs`` re-exports it so
  existing sidecar call sites use the same types)
* ``packages/sie_server/src/sie_server/ipc_types.py`` (Python adapter)

Each declaration is independent — there is no codegen — and every
field travels over msgpack as a named map entry, so a unilateral
rename or new field on either side silently breaks the round-trip on
the other. Run this script to catch that drift before integration.

What we check:

* **Constants** (``IPC_VERSION``, ``METHOD_*``): name and value parity
  across both sources.
* **Struct / class field parity**: for every type whose name appears in
  any of the sources, the set of field names must match across every
  source that declares it. Types that legitimately exist on only one
  side (e.g. ``RequestEnvelope`` is generic-only in Rust and absent
  from Python) are tolerated, but anything declared on both sides
  must agree on field names.
* **Enum variant parity**: same shape as struct fields.

What we deliberately do NOT check:

* **Field types**. Rust ``u32`` ↔ Python ``int`` drift would need a
  per-language type-equivalence table that adds a lot of fragile
  mapping code for no win — type drift surfaces as a msgpack-decode
  error in the integration tests within seconds and the parity tests
  catch the wire-shape end of it. This script guards the
  much-easier-to-miss ``forgot to rename a field`` case.
* **Field order**. msgpack maps are key-addressed; ordering is
  cosmetic.
* **Doc comments**. They differ by intent (Rust uses ``///``, Python
  uses inline ``#``).

Invocation:

    mise exec -- uv run --no-project python tools/check_ipc_types_parity.py

Returns ``0`` on parity, ``1`` on drift with a human-readable diff.
"""

from __future__ import annotations

import ast
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

# ---------------------------------------------------------------------------
# Source locations — single source of truth for the schema files.
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[1]

RUST_PROTOCOL_IPC = REPO_ROOT / "packages" / "sie_server_sidecar" / "src" / "protocol" / "ipc_types.rs"
PY_SERVER_IPC = REPO_ROOT / "packages" / "sie_server" / "src" / "sie_server" / "ipc_types.py"

# Types that are intentionally one-sided. Adding a name to this set
# encodes a deliberate decision; anything not on the list and not
# present on every side is reported as drift.
#
# `RequestEnvelope` is generic-only in Rust (carries the body type
# parameter); the Python server reads request frames as raw dicts in
# `ipc_server._dispatch_frame`, so there is no mirror class to match.
#
# `ResponseEnvelope` IS declared on the Python side (typed) but the
# Rust side carries it generic over the body — both sides agree on the
# named fields, so we keep it in the parity check.
#
# `DrainRequest` is declared in Rust but only via the wire on Python
# (Python reads `deadline_ms` as a dict key in `_handle_drain` and never
# builds a typed struct). Tolerate.
ONE_SIDED_NAMES: frozenset[str] = frozenset(
    {
        "RequestEnvelope",
        "DrainRequest",
        # `RunBatchOp` is a Python `Literal[...]` mirror of the
        # `op` discriminator string. Rust intentionally types `op`
        # as `String` and validates the value at dispatch time, so
        # there is no Rust enum to align against. The valid set is
        # co-tested by the parity fixtures in `tests/parity/`.
        "RunBatchOp",
    }
)


# ---------------------------------------------------------------------------
# Parsed schema model — the shape we normalise each source into so the
# comparator is language-agnostic.
# ---------------------------------------------------------------------------


@dataclass
class TypeDecl:
    """Either a struct/class with field names, or an enum with variant
    names. Stored uniformly: ``fields`` doubles as the variant set for
    enums (msgpack rename_all serialises Rust enums as strings, so the
    Python `Literal[...]` on the other side mirrors the variant set).
    """

    name: str
    kind: str  # "struct" | "enum"
    fields: set[str] = field(default_factory=set)
    location: str = ""  # "<filename>:<line>" for diagnostics


@dataclass
class Schema:
    label: str
    constants: dict[str, str] = field(default_factory=dict)
    types: dict[str, TypeDecl] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Rust parser — regex-based, deliberately scoped to ``ipc_types.rs`` files
# whose shape is uniform (every struct has `#[derive(...)]\npub struct ...`
# and every field is `pub field_name: Type,`). This is not a general-purpose
# Rust parser; if these files start using more exotic syntax, lean on
# `syn` instead.
# ---------------------------------------------------------------------------


def parse_rust(path: Path, label: str) -> Schema:
    src = path.read_text()
    schema = Schema(label=label)

    # Constants: `pub const NAME: TYPE = VALUE;` on a single line.
    # Captures the literal value as a string so we can equality-check
    # numeric and string constants without parsing types.
    for match in re.finditer(
        r"^pub const (?P<name>[A-Z_][A-Z0-9_]*)\s*:\s*[^=]+=\s*(?P<value>.+?);",
        src,
        flags=re.MULTILINE,
    ):
        schema.constants[match.group("name")] = match.group("value").strip()

    # Strip block doc-comments and field-level attributes inside struct
    # bodies before slicing out fields. This keeps the field regex
    # simple — match `pub <name>:` only.
    #
    # We capture preceding `#[...]` attribute lines so we can pluck
    # `#[serde(rename_all = "snake_case")]` off enums and apply the
    # rename to variant names — without it, Rust `Ready` ≠ Python
    # `"ready"` even though they are the SAME wire string.
    type_re = re.compile(
        r"(?P<attrs>(?:#\[[^\]]+\]\s*)*)"
        r"pub (?P<kind>struct|enum) (?P<name>[A-Za-z_][A-Za-z0-9_]*)"
        # optional generics like `<'a, B: Serialize>` — we don't care
        # about their content, just need to skip past them.
        r"(?:<[^>]*>)?"
        r"\s*\{(?P<body>.*?)\n\}",
        flags=re.DOTALL,
    )
    field_re = re.compile(
        r"^\s*pub (?P<name>[A-Za-z_][A-Za-z0-9_]*)\s*:",
        flags=re.MULTILINE,
    )
    enum_variant_re = re.compile(
        r"^\s*(?P<name>[A-Z][A-Za-z0-9_]*)\s*[,({]",
        flags=re.MULTILINE,
    )
    rename_all_re = re.compile(r'#\[serde\([^)]*rename_all\s*=\s*"(?P<style>[^"]+)"[^)]*\)\]')

    for match in type_re.finditer(src):
        name = match.group("name")
        kind = match.group("kind")
        body = match.group("body")
        attrs = match.group("attrs") or ""
        line = src[: match.start()].count("\n") + 1
        decl = TypeDecl(name=name, kind=kind, location=f"{path.name}:{line}")
        if kind == "struct":
            decl.fields = {m.group("name") for m in field_re.finditer(body)}
        else:
            variants = {m.group("name") for m in enum_variant_re.finditer(body)}
            rm = rename_all_re.search(attrs)
            if rm is not None:
                decl.fields = {_apply_serde_rename(v, rm.group("style")) for v in variants}
            else:
                decl.fields = variants
        schema.types[name] = decl

    return schema


def _apply_serde_rename(name: str, style: str) -> str:
    """Mirror serde's `rename_all` for the styles this codebase uses
    (today: `snake_case` only). Adding a new style on either side is a
    one-liner here; we keep the supported set narrow so an unfamiliar
    style fails the check loudly instead of silently letting a rename
    drift through.
    """
    if style == "snake_case":
        # Rust variant names are `UpperCamelCase` (enforced by the
        # compiler). Convert to snake_case the way serde does:
        # insert `_` between an upper run and the next word, then
        # lowercase. `LoadingStarted` → `loading_started`,
        # `RetryLater` → `retry_later`. Acronym-aware logic is
        # unnecessary here — none of the IPC enums use them.
        out: list[str] = []
        for i, ch in enumerate(name):
            if ch.isupper() and i > 0:
                out.append("_")
            out.append(ch.lower())
        return "".join(out)
    raise ValueError(
        f"unsupported serde rename_all style {style!r} on enum (extend "
        "_apply_serde_rename in tools/check_ipc_types_parity.py to teach "
        "the parity check about this style)"
    )


# ---------------------------------------------------------------------------
# Python parser — `ast`-based. Captures top-level constants
# (`Name = value` and `Name: type = value`) plus `msgspec.Struct` /
# `msgspec.Struct(tag=...)` subclasses with their annotated fields,
# plus `Literal[...]` aliases (treated as enum variant sets so Python
# mirror types like `ReadinessState` line up with Rust enums).
# ---------------------------------------------------------------------------


def _is_msgspec_struct_class(node: ast.ClassDef) -> bool:
    """True if `class Foo(msgspec.Struct[...])` — covers both the
    plain inheritance and the tagged variant Python uses for the
    response envelope.
    """
    for base in node.bases:
        if isinstance(base, ast.Attribute) and base.attr == "Struct":
            return True
        if isinstance(base, ast.Name) and base.id == "Struct":
            return True
        if isinstance(base, ast.Call):
            # Defensive: msgspec.Struct used as a metaclass-like
            # expression. Not used in this codebase today but cheap
            # to support.
            if isinstance(base.func, ast.Attribute) and base.func.attr == "Struct":
                return True
    return False


def _literal_variants(node: ast.expr) -> set[str] | None:
    """If `node` is `Literal["a", "b", ...]`, return {"a", "b", ...}
    in their Python wire-string form (kept verbatim — they are the
    msgpack values, not Python identifiers). Otherwise return None
    so the caller skips the binding.
    """
    if not isinstance(node, ast.Subscript):
        return None
    value = node.value
    is_literal = (isinstance(value, ast.Name) and value.id == "Literal") or (
        isinstance(value, ast.Attribute) and value.attr == "Literal"
    )
    if not is_literal:
        return None
    variants: set[str] = set()
    slice_node = node.slice
    elts = slice_node.elts if isinstance(slice_node, ast.Tuple) else [slice_node]
    for elt in elts:
        if isinstance(elt, ast.Constant) and isinstance(elt.value, str):
            variants.add(elt.value)
        else:
            return None
    return variants


def parse_python(path: Path, label: str) -> Schema:
    src = path.read_text()
    tree = ast.parse(src, filename=str(path))
    schema = Schema(label=label)

    for node in tree.body:
        # Top-level constants: METHOD_FOO = "Foo", IPC_VERSION = 1
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            name = node.target.id
            if name.isupper() and node.value is not None:
                schema.constants[name] = ast.unparse(node.value).strip()
        elif isinstance(node, ast.Assign) and len(node.targets) == 1:
            tgt = node.targets[0]
            if isinstance(tgt, ast.Name) and tgt.id.isupper():
                schema.constants[tgt.id] = ast.unparse(node.value).strip()
            # Literal[...] type aliases — the Python mirror for Rust
            # `enum X { ... }` with `#[serde(rename_all="snake_case")]`.
            elif isinstance(tgt, ast.Name) and not tgt.id.isupper():
                variants = _literal_variants(node.value)
                if variants is not None:
                    line = node.lineno
                    schema.types[tgt.id] = TypeDecl(
                        name=tgt.id,
                        kind="enum",
                        fields=variants,
                        location=f"{path.name}:{line}",
                    )

        elif isinstance(node, ast.ClassDef) and _is_msgspec_struct_class(node):
            fields_set: set[str] = set()
            for stmt in node.body:
                # `field_name: type = default` → AnnAssign with target=Name
                if isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name):
                    fields_set.add(stmt.target.id)
                # `field_name: type` (no default) — same shape, also annassign.
            schema.types[node.name] = TypeDecl(
                name=node.name,
                kind="struct",
                fields=fields_set,
                location=f"{path.name}:{node.lineno}",
            )

    return schema


# ---------------------------------------------------------------------------
# Comparator — emits one drift line per disagreement so automated logs are
# easy to scan; returns the number of drift findings so the caller
# can pass it through as the exit code.
# ---------------------------------------------------------------------------


def compare(schemas: list[Schema]) -> list[str]:
    drift: list[str] = []

    # Constants must match in name and value across every schema that
    # declares them. (We allow a constant to be missing on the Python
    # side iff that side is the Python schema and the constant is
    # marked as ignored — none today.) Keep both directions: a Python
    # rename has to agree with a Rust rename and vice versa.
    all_const_names = {n for s in schemas for n in s.constants}
    for cname in sorted(all_const_names):
        present: list[tuple[str, object]] = []
        for s in schemas:
            if cname not in s.constants:
                drift.append(
                    f"[const] {cname!r} is missing from {s.label} but "
                    f"present elsewhere ({', '.join(o.label for o in schemas if cname in o.constants)})."
                )
                continue
            raw_value = s.constants[cname]
            try:
                # Rust and Python spell string literals with different quote
                # styles. Compare their literal values, not source spelling.
                value: object = ast.literal_eval(raw_value)
            except (SyntaxError, ValueError):
                value = raw_value
            present.append((s.label, value))
        if present and any(value != present[0][1] for _, value in present[1:]):
            drift.append(
                f"[const] {cname!r} value drift: " + ", ".join(f"{label}={value!r}" for label, value in present)
            )

    # Type-name parity. We tolerate the names on `ONE_SIDED_NAMES`;
    # everything else must appear on every schema or appear on none.
    all_type_names = {n for s in schemas for n in s.types}
    for tname in sorted(all_type_names):
        if tname in ONE_SIDED_NAMES:
            continue
        present_in = [s for s in schemas if tname in s.types]
        if len(present_in) != len(schemas):
            missing = ", ".join(s.label for s in schemas if tname not in s.types)
            present = ", ".join(s.label for s in present_in)
            drift.append(
                f"[type] {tname!r} present in [{present}] but missing from [{missing}]. "
                f"Add it on the missing side(s) or list it in ONE_SIDED_NAMES with a comment."
            )
            continue

        # Field / variant set must match.
        labelled_decls = [(schema.label, schema.types[tname]) for schema in schemas]
        first_label, first_decl = labelled_decls[0]
        first_fields = first_decl.fields
        for other_label, declaration in labelled_decls[1:]:
            if declaration.fields != first_fields:
                only_first = sorted(first_fields - declaration.fields)
                only_other = sorted(declaration.fields - first_fields)
                drift.append(
                    f"[fields] {tname!r} drift between {first_decl.location} "
                    f"({first_label}) and {declaration.location} ({other_label}):"
                    + (f"\n  only in {first_label}: {only_first}" if only_first else "")
                    + (f"\n  only in {other_label}: {only_other}" if only_other else "")
                )
                # Don't emit a second diff for the same name — first
                # one tells the operator what to fix.
                break

    return drift


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> int:
    schemas = [
        parse_rust(RUST_PROTOCOL_IPC, label="worker-sidecar"),
        parse_python(PY_SERVER_IPC, label="python-server"),
    ]

    drift = compare(schemas)
    if not drift:
        # Print a small breadcrumb so automated logs show that the check ran.
        print(f"ipc-types parity: OK ({len(schemas[0].types)} types, {len(schemas[0].constants)} constants)")
        return 0

    print(
        "ipc-types parity DRIFT detected. The IPC wire contract is declared on both sides and they have diverged:",
        file=sys.stderr,
    )
    print(file=sys.stderr)
    for entry in drift:
        print(f"  - {entry}", file=sys.stderr)
    print(file=sys.stderr)
    print(
        f"Sources:\n  - {RUST_PROTOCOL_IPC.relative_to(REPO_ROOT)}\n  - {PY_SERVER_IPC.relative_to(REPO_ROOT)}",
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
