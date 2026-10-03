#!/usr/bin/env python3
"""Pin the independently deployed response-chunk v1 wire contracts.

The IPC and NATS result transports are separate protocols with separate
negotiation and limit domains. Their declarations intentionally remain local
to independently built peers, so this checker makes v1 drift a validation
failure.

Values are pinned to v1, not merely compared pairwise: changing every current
peer under the same v1 discriminator would still break rolling compatibility
with an older image.
"""

from __future__ import annotations

import ast
import re
import sys
from collections.abc import Mapping
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

IPC_FIELDS = {
    "version",
    "request_id",
    "transfer_digest",
    "chunk_index",
    "chunk_count",
    "total_bytes",
    "payload",
    "kind",
}
RESULT_FIELDS = {
    "kind",
    "work_item_id",
    "request_id",
    "item_index",
    "transfer_digest",
    "chunk_index",
    "chunk_count",
    "total_bytes",
    "payload",
}

IPC_KIND_V1 = "ipc_response_chunk_v1"
IPC_VERSION_V1 = 1
IPC_LEGACY_FRAME_BYTES_V1 = 32 * 1024 * 1024
IPC_CHUNK_PAYLOAD_BYTES_V1 = 4 * 1024 * 1024
IPC_MAX_RESPONSE_BYTES_V1 = 128 * 1024 * 1024
IPC_MAX_CHUNKS_V1 = 64

RESULT_KIND_V1 = "result_chunk_v1"
RESULT_MAX_BYTES_V1 = 16 * 1024 * 1024
RESULT_MAX_CHUNKS_V1 = 64
TRANSFER_DIGEST_BYTES_V1 = 32

IPC_CHUNK_CAPABILITY_V1 = "accepts_ipc_response_chunks_v1"
RESULT_CHUNK_CAPABILITY_V1 = "accepts_result_chunks"

SIDECAR_IPC_TYPES = "packages/sie_server_sidecar/src/protocol/ipc_types.rs"
PYTHON_IPC_TYPES = "packages/sie_server/src/sie_server/ipc_types.py"
PYTHON_IPC_SERVER = "packages/sie_server/src/sie_server/ipc_server.py"
SIDECAR_IPC_CLIENT = "packages/sie_server_sidecar/src/ipc_client.rs"
SIDECAR_IPC_MUX = "packages/sie_server_sidecar/src/ipc_mux.rs"
SIDECAR_IPC_CHUNKS = "packages/sie_server_sidecar/src/protocol/response_chunks.rs"
SIDECAR_WORK_TYPES = "packages/sie_server_sidecar/src/work_types.rs"
SIDECAR_PUBLISHER = "packages/sie_server_sidecar/src/publisher.rs"
GATEWAY_PUBLISHER = "packages/sie_gateway/src/queue/publisher.rs"


def _read(relative_path: str, overrides: Mapping[str, str] | None) -> str:
    if overrides is not None and relative_path in overrides:
        return overrides[relative_path]
    return (REPO_ROOT / relative_path).read_text()


def _eval_int_expr(expr: str) -> int:
    node = ast.parse(expr.strip(), mode="eval").body

    def evaluate(current: ast.expr) -> int:
        if isinstance(current, ast.Constant) and type(current.value) is int:
            return current.value
        if isinstance(current, ast.UnaryOp) and isinstance(current.op, ast.USub):
            return -evaluate(current.operand)
        if isinstance(current, ast.BinOp):
            left = evaluate(current.left)
            right = evaluate(current.right)
            if isinstance(current.op, ast.Mult):
                return left * right
            if isinstance(current.op, ast.Add):
                return left + right
            if isinstance(current.op, ast.Sub):
                return left - right
            if isinstance(current.op, ast.LShift):
                return left << right
        raise ValueError(f"unsupported integer constant expression: {expr!r}")

    return evaluate(node)


def _rust_const(source: str, name: str) -> str:
    match = re.search(
        rf"^(?:pub(?:\([^)]*\))?\s+)?const\s+{re.escape(name)}\s*:\s*[^=]+\s*=\s*(?P<value>[^;]+);",
        source,
        flags=re.MULTILINE,
    )
    if match is None:
        raise ValueError(f"Rust constant {name} not found")
    return match.group("value").strip()


def _python_const(source: str, name: str) -> str:
    tree = ast.parse(source)
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target = node.targets[0]
            if isinstance(target, ast.Name) and target.id == name:
                return ast.unparse(node.value)
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            if node.target.id == name and node.value is not None:
                return ast.unparse(node.value)
    raise ValueError(f"Python constant {name} not found")


def _rust_struct_body(source: str, name: str) -> str:
    match = re.search(
        rf"(?:#\[[^\]]+\]\s*)*(?:pub(?:\([^)]*\))?\s+)?struct\s+{re.escape(name)}"
        rf"(?:<[^>]*>)?\s*\{{(?P<body>.*?)\n\}}",
        source,
        flags=re.DOTALL,
    )
    if match is None:
        raise ValueError(f"Rust struct {name} not found")
    return match.group("body")


def _rust_struct_fields(source: str, name: str) -> set[str]:
    return {
        field.group("name")
        for field in re.finditer(
            r"^\s*(?:pub\s+)?(?P<name>[a-z_][a-z0-9_]*)\s*:",
            _rust_struct_body(source, name),
            flags=re.MULTILINE,
        )
    }


def _rust_field_attributes(source: str, struct_name: str, field_name: str) -> str:
    body = _rust_struct_body(source, struct_name)
    match = re.search(
        rf"(?P<attrs>(?:\s*#\[[^\]]+\]\s*)*)\s*(?:pub\s+)?\b{re.escape(field_name)}\s*:",
        body,
    )
    if match is None:
        raise ValueError(f"Rust field {struct_name}.{field_name} not found")
    return re.sub(r"\s+", "", match.group("attrs"))


def _rust_field_has_serde_option(source: str, struct_name: str, field_name: str, option: str) -> bool:
    attributes = _rust_field_attributes(source, struct_name, field_name)
    return "#[serde(" in attributes and option in attributes


def _python_struct_fields(source: str, name: str) -> set[str]:
    tree = ast.parse(source)
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == name:
            return {
                statement.target.id
                for statement in node.body
                if isinstance(statement, ast.AnnAssign) and isinstance(statement.target, ast.Name)
            }
    raise ValueError(f"Python class {name} not found")


def _python_field_annotation(source: str, class_name: str, field_name: str) -> str:
    tree = ast.parse(source)
    for node in tree.body:
        if not isinstance(node, ast.ClassDef) or node.name != class_name:
            continue
        for statement in node.body:
            if (
                isinstance(statement, ast.AnnAssign)
                and isinstance(statement.target, ast.Name)
                and statement.target.id == field_name
            ):
                return ast.unparse(statement.annotation)
    raise ValueError(f"Python field annotation {class_name}.{field_name} not found")


def _python_has_exact_true_lookup(source: str, key: str) -> bool:
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if not (
            isinstance(node, ast.Compare)
            and len(node.ops) == 1
            and isinstance(node.ops[0], ast.Is)
            and len(node.comparators) == 1
            and isinstance(node.comparators[0], ast.Constant)
            and node.comparators[0].value is True
            and isinstance(node.left, ast.Call)
            and isinstance(node.left.func, ast.Attribute)
            and node.left.func.attr == "get"
            and len(node.left.args) == 1
            and isinstance(node.left.args[0], ast.Constant)
            and node.left.args[0].value == key
        ):
            continue
        return True
    return False


def _python_field_default(source: str, class_name: str, field_name: str) -> object:
    tree = ast.parse(source)
    for node in tree.body:
        if not isinstance(node, ast.ClassDef) or node.name != class_name:
            continue
        for statement in node.body:
            if (
                isinstance(statement, ast.AnnAssign)
                and isinstance(statement.target, ast.Name)
                and statement.target.id == field_name
                and statement.value is not None
            ):
                return ast.literal_eval(statement.value)
    raise ValueError(f"Python field default {class_name}.{field_name} not found")


def _check(errors: list[str], label: str, actual: object, expected: object) -> None:
    if actual != expected:
        errors.append(f"{label}: expected {expected!r}, found {actual!r}")


def _check_source_pattern(errors: list[str], label: str, source: str, pattern: str) -> None:
    _check(
        errors,
        label,
        re.search(pattern, source, flags=re.MULTILINE | re.DOTALL) is not None,
        True,
    )


_RUST_CFG_TEST_ATTRIBUTE = re.compile(
    r"^[ \t]*#\s*\[\s*cfg\s*\(\s*test\s*\)\s*\]",
    flags=re.MULTILINE,
)
_RUST_ITEM_TRIVIA = r"(?:(?:[ \t\r\n]+)|(?://[^\r\n]*(?:\r?\n|$))|(?:/\*.*?\*/))*"
_RUST_ADDITIONAL_ATTRIBUTES = rf"(?:#\[[^\]\r\n]*\]{_RUST_ITEM_TRIVIA})*"
_RUST_TEST_MODULE_ITEM = re.compile(
    rf"^{_RUST_ITEM_TRIVIA}{_RUST_ADDITIONAL_ATTRIBUTES}"
    r"(?:(?:pub(?:\s*\([^)]*\))?)[ \t\r\n]+)?"
    r"mod[ \t\r\n]+(?:r#)?[A-Za-z_][A-Za-z0-9_]*[ \t\r\n]*\{",
    flags=re.DOTALL,
)
_RUST_MODULE_ITEM_PREFIX = re.compile(
    rf"^{_RUST_ITEM_TRIVIA}{_RUST_ADDITIONAL_ATTRIBUTES}"
    r"(?:(?:pub(?:\s*\([^)]*\))?)[ \t\r\n]+)?mod\b",
    flags=re.DOTALL,
)
_RUST_TEST_HELPER_METHOD = re.compile(
    rf"^{_RUST_ITEM_TRIVIA}{_RUST_ADDITIONAL_ATTRIBUTES}"
    r"(?:(?:pub(?:\s*\([^)]*\))?)[ \t\r\n]+)?"
    r"(?:(?:async|const|unsafe)[ \t\r\n]+)*"
    r"fn[ \t\r\n]+(?:r#)?[A-Za-z_][A-Za-z0-9_]*\b[^;{}]*\{",
    flags=re.DOTALL,
)
_RUST_RAW_STRING_START = re.compile(r'(?:b)?r(?P<hashes>#{0,255})"')
_RUST_CHAR_LITERAL = re.compile(r"'(?:\\(?:x[0-9A-Fa-f]{2}|u\{[0-9A-Fa-f_]{1,6}\}|.)|[^\\'\r\n])'")


def _rust_non_code_end(source: str, index: int) -> int | None:
    if source.startswith("//", index):
        newline = source.find("\n", index + 2)
        return len(source) if newline < 0 else newline + 1
    if source.startswith("/*", index):
        comment_depth = 1
        cursor = index + 2
        while cursor < len(source) and comment_depth:
            if source.startswith("/*", cursor):
                comment_depth += 1
                cursor += 2
            elif source.startswith("*/", cursor):
                comment_depth -= 1
                cursor += 2
            else:
                cursor += 1
        if comment_depth:
            raise ValueError("Rust source has an unterminated block comment")
        return cursor
    raw_string = _RUST_RAW_STRING_START.match(source, index)
    if raw_string:
        terminator = f'"{raw_string.group("hashes")}'
        end = source.find(terminator, raw_string.end())
        if end < 0:
            raise ValueError("Rust source has an unterminated raw string")
        return end + len(terminator)
    if source[index] == '"':
        cursor = index + 1
        while cursor < len(source):
            if source[cursor] == "\\":
                cursor += 2
            elif source[cursor] == '"':
                return cursor + 1
            else:
                cursor += 1
        raise ValueError("Rust source has an unterminated string")
    char_literal = _RUST_CHAR_LITERAL.match(source, index)
    return char_literal.end() if char_literal else None


def _rust_code_mask(source: str) -> str:
    """Blank comments and literals while preserving source offsets and lines."""
    masked = list(source)
    index = 0
    while index < len(source):
        if (non_code_end := _rust_non_code_end(source, index)) is None:
            index += 1
            continue
        for cursor in range(index, non_code_end):
            if masked[cursor] not in "\r\n":
                masked[cursor] = " "
        index = non_code_end
    return "".join(masked)


def _rust_nesting_depth(source: str, stop: int) -> int | None:
    depth = 0
    index = 0
    while index < stop:
        if (non_code_end := _rust_non_code_end(source, index)) is not None:
            if non_code_end > stop:
                return None
            index = non_code_end
            continue
        if source[index] == "{":
            depth += 1
        elif source[index] == "}":
            depth -= 1
            if depth < 0:
                raise ValueError("Rust source has an unmatched closing brace")
        index += 1
    return depth


def _rust_braced_item_end(source: str, opening_brace: int) -> int:
    depth = 0
    index = opening_brace
    while index < len(source):
        if (non_code_end := _rust_non_code_end(source, index)) is not None:
            index = non_code_end
            continue
        if source[index] == "{":
            depth += 1
        elif source[index] == "}":
            depth -= 1
            if depth == 0:
                return index + 1
        index += 1
    raise ValueError("Rust cfg(test) helper has an unterminated body")


def _blank_rust_ranges(source: str, ranges: list[tuple[int, int]]) -> str:
    result = source
    for start, end in reversed(ranges):
        blank = "".join("\n" if character == "\n" else " " for character in result[start:end])
        result = result[:start] + blank + result[end:]
    return result


def _rust_without_nested_test_helpers(source: str) -> str:
    helper_ranges: list[tuple[int, int]] = []
    masked_source = _rust_code_mask(source)
    for attribute in _RUST_CFG_TEST_ATTRIBUTE.finditer(masked_source):
        if any(start <= attribute.start() < end for start, end in helper_ranges):
            continue
        nesting_depth = _rust_nesting_depth(source, attribute.start())
        if nesting_depth is None:
            continue
        following = masked_source[attribute.end() :]
        if nesting_depth > 0 and (helper := _RUST_TEST_HELPER_METHOD.match(following)):
            # A few scanned impls expose test-only helpers before their terminal
            # test module. Remove only a lexically balanced nested method before
            # the strict production/test-module split runs.
            opening_brace = attribute.end() + helper.end() - 1
            helper_ranges.append((attribute.start(), _rust_braced_item_end(source, opening_brace)))
    return _blank_rust_ranges(source, helper_ranges)


def _rust_production(source: str) -> str:
    boundaries: list[int] = []
    masked_source = _rust_code_mask(source)
    for attribute in _RUST_CFG_TEST_ATTRIBUTE.finditer(masked_source):
        nesting_depth = _rust_nesting_depth(source, attribute.start())
        if nesting_depth is None:
            continue
        following = masked_source[attribute.end() :]
        if _RUST_TEST_MODULE_ITEM.match(following):
            if nesting_depth:
                raise ValueError("Rust cfg(test) module boundary is nested and unsupported")
            boundaries.append(attribute.start())
        elif _RUST_MODULE_ITEM_PREFIX.match(following):
            raise ValueError("Rust cfg(test) module boundary is malformed or unsupported")
        else:
            raise ValueError("Rust cfg(test) item is not a recognized inline module boundary")

    if len(boundaries) > 1:
        raise ValueError("Rust cfg(test) module boundary is ambiguous: multiple modules found")
    return source[: boundaries[0]] if boundaries else source


def _rust_production_source(source: str) -> str:
    return _rust_production(_rust_without_nested_test_helpers(source))


def validate_protocol(overrides: Mapping[str, str] | None = None) -> list[str]:
    errors: list[str] = []

    ipc_struct_sites = (
        ("IPC sidecar envelope", SIDECAR_IPC_TYPES, "rust"),
        ("IPC Python envelope", PYTHON_IPC_TYPES, "python"),
    )
    for label, path, language in ipc_struct_sites:
        source = _read(path, overrides)
        fields = (
            _python_struct_fields(source, "IpcResponseChunkV1")
            if language == "python"
            else _rust_struct_fields(source, "IpcResponseChunkV1")
        )
        _check(errors, f"{label} fields", fields, IPC_FIELDS)

    sidecar_producer_fields = _rust_struct_fields(_read(SIDECAR_PUBLISHER, overrides), "ResultChunkV1Ref")
    _check(errors, "NATS sidecar producer fields", sidecar_producer_fields, RESULT_FIELDS)

    result_struct_sites = (
        ("NATS sidecar envelope", SIDECAR_WORK_TYPES),
        ("NATS gateway envelope", GATEWAY_PUBLISHER),
    )
    for label, path in result_struct_sites:
        fields = _rust_struct_fields(_read(path, overrides), "ResultChunkV1")
        _check(errors, f"{label} fields", fields, RESULT_FIELDS)

    for label, path, struct_name in (
        ("IPC sidecar envelope", SIDECAR_IPC_TYPES, "IpcResponseChunkV1"),
        ("NATS sidecar envelope", SIDECAR_WORK_TYPES, "ResultChunkV1"),
        ("NATS gateway envelope", GATEWAY_PUBLISHER, "ResultChunkV1"),
        ("NATS sidecar producer", SIDECAR_PUBLISHER, "ResultChunkV1Ref"),
    ):
        source = _read(path, overrides)
        for field_name in ("transfer_digest", "payload"):
            _check(
                errors,
                f'{label} {field_name} serde(with = "serde_bytes")',
                _rust_field_has_serde_option(
                    source,
                    struct_name,
                    field_name,
                    'with="serde_bytes"',
                ),
                True,
            )

    python_ipc_source = _read(PYTHON_IPC_TYPES, overrides)
    for field_name in ("transfer_digest", "payload"):
        _check(
            errors,
            f"IPC Python envelope {field_name} annotation",
            _python_field_annotation(python_ipc_source, "IpcResponseChunkV1", field_name),
            "bytes",
        )

    # `serde_bytes`/`bytes` only pins the MessagePack representation. Pin the
    # actual v1 integrity semantics too: producers hash the complete logical
    # result with SHA-256, consumers hash the reassembled payload and compare
    # it, and every peer enforces SHA-256's exact 32-byte output length.
    python_ipc_producer = _read(PYTHON_IPC_SERVER, overrides)
    _check_source_pattern(
        errors,
        "IPC Python producer computes SHA-256 over the complete response",
        python_ipc_producer,
        r"^[ \t]*digest[ \t]*=[ \t]*hashlib\.sha256\([ \t]*payload[ \t]*\)"
        r"\.digest\([ \t]*\)[ \t]*$",
    )
    _check_source_pattern(
        errors,
        "IPC Python producer publishes the computed SHA-256 digest",
        python_ipc_producer,
        r"\btransfer_digest\s*=\s*digest\b",
    )

    sidecar_ipc_consumer = _rust_production_source(_read(SIDECAR_IPC_CHUNKS, overrides))
    _check_source_pattern(
        errors,
        "IPC sidecar consumer requires an exact 32-byte digest",
        sidecar_ipc_consumer,
        rf"\bchunk\.transfer_digest\.len\(\s*\)\s*!=\s*{TRANSFER_DIGEST_BYTES_V1}\b",
    )
    _check_source_pattern(
        errors,
        "IPC sidecar consumer stores an exact 32-byte digest",
        sidecar_ipc_consumer,
        rf"\blet\s+digest\s*:\s*\[\s*u8\s*;\s*{TRANSFER_DIGEST_BYTES_V1}\s*\]"
        r"\s*=\s*chunk\s*\.transfer_digest\s*\.as_slice\(\s*\)\s*\.try_into\(\s*\)",
    )
    _check_source_pattern(
        errors,
        "IPC sidecar consumer verifies reassembled bytes with SHA-256",
        sidecar_ipc_consumer,
        rf"\blet\s+actual_digest\s*:\s*\[\s*u8\s*;\s*{TRANSFER_DIGEST_BYTES_V1}\s*\]"
        r"\s*=\s*completed\.hasher\.finalize\(\s*\)\.into\(\s*\)\s*;"
        r"\s*if\s+actual_digest\s*!=\s*completed\.digest",
    )
    _check_source_pattern(
        errors,
        "IPC sidecar consumer hashes each payload with SHA-256",
        sidecar_ipc_consumer,
        r"\bhasher\s*:\s*Sha256\b.*\bhasher\s*:\s*Sha256::new\(\s*\)"
        r".*\bstate\.hasher\.update\(\s*&chunk\.payload\s*\)",
    )

    sidecar_nats_producer = _rust_production_source(_read(SIDECAR_PUBLISHER, overrides))
    _check_source_pattern(
        errors,
        "NATS sidecar producer computes SHA-256 over the complete result",
        sidecar_nats_producer,
        rf"\blet\s+transfer_digest\s*:\s*\[\s*u8\s*;\s*{TRANSFER_DIGEST_BYTES_V1}\s*\]"
        r"\s*=\s*Sha256::digest\(\s*&result_bytes\s*\)\.into\(\s*\)",
    )
    _check_source_pattern(
        errors,
        "NATS sidecar producer publishes the computed SHA-256 digest",
        sidecar_nats_producer,
        r"\btransfer_digest\s*:\s*&transfer_digest\b",
    )

    gateway_nats_consumer = _rust_production_source(_read(GATEWAY_PUBLISHER, overrides))
    _check_source_pattern(
        errors,
        "NATS gateway consumer requires an exact 32-byte digest",
        gateway_nats_consumer,
        rf"\blet\s+transfer_digest\s*:\s*\[\s*u8\s*;\s*{TRANSFER_DIGEST_BYTES_V1}\s*\]"
        r"\s*=\s*chunk\s*\.transfer_digest\s*\.as_slice\(\s*\)\s*\.try_into\(\s*\)",
    )
    _check_source_pattern(
        errors,
        "NATS gateway consumer verifies reassembled bytes with SHA-256",
        gateway_nats_consumer,
        r"\blet\s+mut\s+transfer_hasher\s*=\s*Sha256::new\(\s*\)\s*;"
        r".*\btransfer_hasher\.update\(\s*&payload\s*\)"
        r".*\blet\s+actual_digest\s*=\s*transfer_hasher\.finalize\(\s*\)\s*;"
        r"\s*if\s+actual_digest\[\.\.\]\s*!=\s*partial\.transfer_digest",
    )

    for label, path, struct_name, capability in (
        ("IPC sidecar request", SIDECAR_IPC_TYPES, "RequestEnvelope", IPC_CHUNK_CAPABILITY_V1),
        ("NATS gateway work item", GATEWAY_PUBLISHER, "WorkItemRef", RESULT_CHUNK_CAPABILITY_V1),
        ("NATS sidecar work item", SIDECAR_WORK_TYPES, "WorkItem", RESULT_CHUNK_CAPABILITY_V1),
    ):
        fields = _rust_struct_fields(_read(path, overrides), struct_name)
        _check(errors, f"{label} capability field", capability in fields, True)

    _check(
        errors,
        "NATS sidecar work item capability defaults false when absent",
        _rust_field_has_serde_option(
            _read(SIDECAR_WORK_TYPES, overrides),
            "WorkItem",
            RESULT_CHUNK_CAPABILITY_V1,
            "default",
        ),
        True,
    )

    _check(
        errors,
        "IPC Python capability requires exact true",
        _python_has_exact_true_lookup(_read(PYTHON_IPC_SERVER, overrides), IPC_CHUNK_CAPABILITY_V1),
        True,
    )

    sidecar_client_production = _rust_production_source(_read(SIDECAR_IPC_CLIENT, overrides))
    _check(
        errors,
        "IPC sidecar advertises chunk capability",
        len(re.findall(rf"\b{IPC_CHUNK_CAPABILITY_V1}\s*:\s*true\b", sidecar_client_production)),
        1,
    )
    gateway_production = _rust_production_source(_read(GATEWAY_PUBLISHER, overrides))
    _check(
        errors,
        "NATS gateway advertises chunk capability on every producer",
        len(re.findall(rf"\b{RESULT_CHUNK_CAPABILITY_V1}\s*:\s*true\b", gateway_production)),
        3,
    )

    for label, path, language in ipc_struct_sites:
        source = _read(path, overrides)
        raw = _python_const(source, "IPC_VERSION") if language == "python" else _rust_const(source, "IPC_VERSION")
        _check(errors, f"{label} IPC_VERSION", _eval_int_expr(raw), IPC_VERSION_V1)

    _check(
        errors,
        "IPC Python kind",
        _python_field_default(_read(PYTHON_IPC_TYPES, overrides), "IpcResponseChunkV1", "kind"),
        IPC_KIND_V1,
    )
    for label, path, name in (
        ("IPC sidecar kind", SIDECAR_IPC_CHUNKS, "IPC_RESPONSE_CHUNK_KIND_V1"),
        ("NATS sidecar kind", SIDECAR_PUBLISHER, "RESULT_CHUNK_KIND"),
        ("NATS gateway kind", GATEWAY_PUBLISHER, "RESULT_CHUNK_KIND"),
    ):
        expected = IPC_KIND_V1 if label.startswith("IPC") else RESULT_KIND_V1
        _check(errors, label, ast.literal_eval(_rust_const(_read(path, overrides), name)), expected)

    for label, path, name, language in (
        ("IPC Python legacy frame", PYTHON_IPC_SERVER, "_MAX_FRAME_BYTES", "python"),
        ("IPC sidecar pool frame", SIDECAR_IPC_CLIENT, "MAX_FRAME_BYTES", "rust"),
        ("IPC sidecar mux frame", SIDECAR_IPC_MUX, "MAX_FRAME_BYTES", "rust"),
        (
            "IPC sidecar reassembly legacy frame",
            SIDECAR_IPC_CHUNKS,
            "LEGACY_IPC_RESPONSE_FRAME_BYTES",
            "rust",
        ),
    ):
        source = _read(path, overrides)
        raw = _python_const(source, name) if language == "python" else _rust_const(source, name)
        _check(errors, label, _eval_int_expr(raw), IPC_LEGACY_FRAME_BYTES_V1)

    for expected, sites in (
        (
            IPC_CHUNK_PAYLOAD_BYTES_V1,
            (
                ("IPC Python chunk payload", PYTHON_IPC_SERVER, "_IPC_RESPONSE_CHUNK_PAYLOAD_BYTES", "python"),
                ("IPC sidecar chunk payload", SIDECAR_IPC_CHUNKS, "IPC_RESPONSE_CHUNK_PAYLOAD_BYTES", "rust"),
            ),
        ),
        (
            IPC_MAX_RESPONSE_BYTES_V1,
            (
                ("IPC Python maximum response", PYTHON_IPC_SERVER, "_MAX_CHUNKED_IPC_RESPONSE_BYTES", "python"),
                ("IPC sidecar maximum response", SIDECAR_IPC_CHUNKS, "MAX_CHUNKED_IPC_RESPONSE_BYTES", "rust"),
            ),
        ),
        (
            IPC_MAX_CHUNKS_V1,
            (
                ("IPC Python maximum chunks", PYTHON_IPC_SERVER, "_MAX_IPC_RESPONSE_CHUNKS", "python"),
                ("IPC sidecar maximum chunks", SIDECAR_IPC_CHUNKS, "MAX_IPC_RESPONSE_CHUNKS", "rust"),
            ),
        ),
        (
            RESULT_MAX_BYTES_V1,
            (
                ("NATS sidecar maximum result", SIDECAR_PUBLISHER, "MAX_CHUNKED_RESULT_BYTES", "rust"),
                ("NATS gateway maximum result", GATEWAY_PUBLISHER, "MAX_RESULT_CHUNK_ITEM_BYTES", "rust"),
            ),
        ),
        (
            RESULT_MAX_CHUNKS_V1,
            (
                ("NATS sidecar maximum chunks", SIDECAR_PUBLISHER, "MAX_RESULT_CHUNKS", "rust"),
                ("NATS gateway maximum chunks", GATEWAY_PUBLISHER, "MAX_RESULT_CHUNKS_PER_ITEM", "rust"),
            ),
        ),
    ):
        for label, path, name, language in sites:
            source = _read(path, overrides)
            raw = _python_const(source, name) if language == "python" else _rust_const(source, name)
            _check(errors, label, _eval_int_expr(raw), expected)

    return errors


def main() -> int:
    errors = validate_protocol()
    if not errors:
        print("response-chunk protocol v1: OK (IPC 2 peers, NATS 2 peers)")
        return 0
    print("response-chunk protocol v1 DRIFT detected:", file=sys.stderr)
    for error in errors:
        print(f"  - {error}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
