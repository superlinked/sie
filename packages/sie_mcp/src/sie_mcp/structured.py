"""Structured-output jobs for ``extract_structured`` and ``generate_structured``.

Both produce schema/grammar-constrained output off the cluster via the
OpenAI-compatible chat-completions surface (``response_format``), *not* the
SIE-native ``generate()`` / ``extract()`` primitives — ``extract.output_schema``
is a declared-but-stub param on every adapter, so grounded JSON comes from a
generative model constrained to the schema and fed the document content.

The cluster (gateway + Outlines backend) is the final authority on grammar
validity; this module pre-validates the JSON-Schema subset (keywords, depth,
node count, and serialized size — mirroring the gateway safety caps) and
rejects EBNF-on-Outlines up front so callers get a clear, fast error instead
of a ``grammar_capability`` / ``unsupported_field`` round-trip.

It also *post-validates* the cluster's response against the requested schema.
Decode-time grammar enforcement is the primary guarantee, but a serving-layer
profile can bypass the grammar FSM (for example, speculative decoding can
produce extra keys or truncated JSON); when that happens the output is rejected with a clear error
rather than returned as if it conformed.
"""

import json
from typing import Any, Protocol

# JSON-Schema keywords Outlines does not support (or compiles at prohibitive
# cost). Mirrors the gateway reject-list in
# ``packages/sie_gateway/src/handlers/grammar.rs`` so a schema accepted here is
# also accepted there. ``$ref`` covers recursion: a recursive JSON Schema can
# only refer back to itself through ``$ref``, so rejecting all ``$ref`` rejects
# recursion too. ``if`` / ``then`` / ``else`` cover conditionals.
_UNSUPPORTED_KEYWORDS = (
    "$ref",
    "$dynamicRef",
    "if",
    "then",
    "else",
    "unevaluatedProperties",
    "dependentSchemas",
)

# Keys whose values are nested sub-schemas; descending into them increments the
# depth counter. Mirrors ``is_schema_nesting_key`` in the gateway.
_SCHEMA_NESTING_KEYS = frozenset(
    {
        "properties",
        "patternProperties",
        "additionalProperties",
        "unevaluatedProperties",
        "items",
        "prefixItems",
        "contains",
        "propertyNames",
        "oneOf",
        "anyOf",
        "allOf",
        "not",
        "definitions",
        "$defs",
        "dependentSchemas",
        "if",
        "then",
        "else",
    }
)

# Mirrors ``MAX_SCHEMA_DEPTH`` in the gateway. A schema at exactly this depth is
# accepted; deeper is rejected.
_MAX_SCHEMA_DEPTH = 16

# The remaining gateway safety caps (``packages/sie_gateway/src/handlers/grammar.rs``),
# mirrored so a schema/grammar accepted at this edge is also accepted there —
# the edge preflight stays a faithful fast-fail, not a looser gate that lets a
# request through only for the gateway to reject it after a round-trip.
#   * ``_MAX_SCHEMA_NODES`` bounds a *wide* schema (depth alone does not stop a
#     shallow object with hundreds of thousands of trivial properties).
#   * ``_MAX_GRAMMAR_BYTES`` bounds the serialized schema size.
#   * ``_MAX_REGEX_LEN`` / ``_MAX_EBNF_LEN`` bound the regex / EBNF source.
_MAX_SCHEMA_NODES = 16 * 1024
_MAX_GRAMMAR_BYTES = 64 * 1024
_MAX_REGEX_LEN = 4 * 1024
_MAX_EBNF_LEN = 8 * 1024

# Models whose generation profile is Outlines-backed and therefore advertise
# only ``json_schema`` + ``regex`` (EBNF/grammar is an upstream SGLang/Outlines
# limitation). An EBNF request to one of these is rejected up front. Both 4B
# generation profiles served on the test cluster (``grammar_backend: outlines``)
# seed the set.
_OUTLINES_BACKED_MODELS = frozenset(
    {
        "Qwen/Qwen3.5-4B",
        "Qwen/Qwen3-4B-Instruct-2507",
    }
)

_EXTRACT_SYSTEM = (
    "You extract structured data from the supplied document content. Use only "
    "information present in the content — do not invent or infer values it does "
    "not support. Where the schema permits and a value is absent, use null."
)
_DEFAULT_EXTRACT_INSTRUCTION = "Extract the structured data described by the schema from the document below."


class StructuredOutputError(Exception):
    """Raised when an input schema/grammar is out of the supported subset.

    Carries a caller-facing message naming the offending key path so the
    schema can be fixed without a cluster round-trip.
    """


class ChatClient(Protocol):
    """The slice of ``SIEAsyncClient`` these jobs need (keeps them unit-testable)."""

    async def chat_completions(self, model: str, messages: Any, **kwargs: Any) -> Any:
        """Run an OpenAI-compatible chat completion against the SIE cluster."""


def _is_outlines_backed(model: str) -> bool:
    return model in _OUTLINES_BACKED_MODELS


def _schema_child_context(parent: str, key: str) -> str:
    """Map a parent schema context + child key to the child's context.

    Mirrors ``schema_child_context`` in the gateway
    (``packages/sie_gateway/src/handlers/grammar.rs``) and
    ``_schema_child_context`` in the server (``api/generate.py``). Keywords are
    only meaningful in a ``"schema"`` position; a ``properties`` /
    ``patternProperties`` / ``$defs`` map has *property names* for keys
    (``"schema_map"``), so a property literally named ``if`` / ``then`` / ``$ref``
    is data, not a keyword.
    """
    if parent == "schema":
        if key in {"properties", "patternProperties", "$defs", "definitions", "dependentSchemas"}:
            return "schema_map"
        if key in {"oneOf", "anyOf", "allOf", "prefixItems"}:
            return "schema_array"
        if key in {"items", "additionalProperties", "contains", "propertyNames", "not", "if", "then", "else"}:
            return "schema"
        return "other"
    if parent == "schema_map":
        return "schema"
    return "other"


def _walk_schema(node: Any, *, path: str, depth: int, counter: list[int], context: str = "schema") -> None:
    # Mirror the gateway's per-node accounting: every visited value (objects,
    # array elements, scalars) counts toward ``_MAX_SCHEMA_NODES`` so a wide
    # schema is bounded even when it stays shallow.
    counter[0] += 1
    if counter[0] > _MAX_SCHEMA_NODES:
        msg = f"output_schema node count exceeds limit ({_MAX_SCHEMA_NODES}) at {path}"
        raise StructuredOutputError(msg)
    if depth > _MAX_SCHEMA_DEPTH:
        msg = f"output_schema nesting depth exceeds limit ({_MAX_SCHEMA_DEPTH}) at {path}"
        raise StructuredOutputError(msg)
    if isinstance(node, dict):
        # Keywords are only rejected in an actual schema position. Under a
        # ``properties`` map (``"schema_map"``) the keys are property *names*
        # supplied by the caller, so a property named ``if`` / ``then`` / ``$ref``
        # is data and must not be treated as a keyword, matching the same
        # context-sensitive validation in the gateway and server.
        if context == "schema":
            # Reject before descending so the message names the shallowest occurrence.
            for kw in _UNSUPPORTED_KEYWORDS:
                if kw in node:
                    detail = (
                        "'$ref' is not supported (the schema subset rejects all $ref, "
                        "including internal '#/...', so recursion is not expressible)"
                        if kw == "$ref"
                        else f"JSON Schema keyword '{kw}' is not supported by the Outlines subset"
                    )
                    raise StructuredOutputError(f"{detail} at {path}.{kw}")
        for key, child in node.items():
            child_depth = depth + 1 if context == "schema" and key in _SCHEMA_NESTING_KEYS else depth
            _walk_schema(
                child,
                path=f"{path}.{key}",
                depth=child_depth,
                counter=counter,
                context=_schema_child_context(context, key),
            )
    elif isinstance(node, list):
        child_context = "schema" if context in {"schema", "schema_array"} else "other"
        for i, child in enumerate(node):
            _walk_schema(child, path=f"{path}[{i}]", depth=depth, counter=counter, context=child_context)


def validate_output_schema(schema: Any) -> None:
    """Enforce the Outlines-supported JSON-Schema subset and the gateway caps.

    Rejects ``$ref`` (and thus recursion), conditionals (``if``/``then``/
    ``else``), other unsupported keywords, nesting deeper than
    :data:`_MAX_SCHEMA_DEPTH`, more than :data:`_MAX_SCHEMA_NODES` nodes, and a
    serialized schema larger than :data:`_MAX_GRAMMAR_BYTES` — mirroring the
    gateway so an edge-accepted schema is gateway-accepted too. Raises
    :class:`StructuredOutputError` on the first violation; returns ``None`` when
    the schema is in-subset.
    """
    if not isinstance(schema, dict):
        msg = "output_schema must be a JSON object"
        raise StructuredOutputError(msg)
    # Serialized-size cap first (mirrors the gateway, which checks payload size
    # before walking) so a pathologically large schema is rejected cheaply.
    serialized_len = len(json.dumps(schema).encode("utf-8"))
    if serialized_len > _MAX_GRAMMAR_BYTES:
        msg = f"output_schema serialized size {serialized_len} bytes exceeds limit ({_MAX_GRAMMAR_BYTES} bytes)"
        raise StructuredOutputError(msg)
    _walk_schema(schema, path="output_schema", depth=0, counter=[0])


def _validate_against_schema(data: Any, schema: dict[str, Any]) -> None:
    """Post-validate cluster output against the requested schema (defense in depth).

    Decode-time grammar enforcement is the primary guarantee, but a serving
    profile can bypass the grammar FSM (speculative decoding can produce extra
    keys or truncated JSON). When the returned
    object does not conform, raise rather than hand back non-conforming JSON as
    if it were schema-valid.

    ``jsonschema`` is a guaranteed transitive dependency of the MCP edge
    (``mcp`` → ``jsonschema``); it is imported lazily to support lightweight
    standalone validation without installing the package. If it is somehow
    absent the check is skipped — this layer is additive safety, not the only
    guard.
    """
    try:
        import jsonschema  # noqa: PLC0415  (lazy: keeps structured.py standalone-importable)
    except ImportError:
        return
    try:
        jsonschema.validate(instance=data, schema=schema)
    except jsonschema.ValidationError as exc:
        msg = (
            "cluster output does not conform to the requested schema — the serving "
            "profile may have bypassed grammar enforcement (e.g. speculative decoding "
            f"on the default Qwen3.5 profile): {exc.message}"
        )
        raise StructuredOutputError(msg) from exc
    except jsonschema.SchemaError:
        # The schema passed our subset walk but jsonschema's metaschema is
        # stricter; don't fail the request on validator strictness — the cluster
        # is the authority on what it can compile.
        return


def _message_content(resp: Any) -> str:
    choices = resp.get("choices") if isinstance(resp, dict) else None
    if not choices:
        msg = "cluster returned no choices"
        raise StructuredOutputError(msg)
    # Defensive against a malformed response shape (e.g. ``choices: [null]`` or a
    # null ``message``): surface a clear typed error, not a raw AttributeError.
    first = choices[0]
    message = first.get("message") if isinstance(first, dict) else None
    content = message.get("content") if isinstance(message, dict) else None
    if not isinstance(content, str):
        msg = "cluster returned no message content"
        raise StructuredOutputError(msg)
    return content


async def extract_structured(
    client: ChatClient,
    *,
    content: str,
    output_schema: dict[str, Any],
    instruction: str | None = None,
    model: str,
    gpu: str | None = None,
    max_completion_tokens: int | None = None,
) -> Any:
    """Extract schema-valid JSON grounded in ``content`` via constrained generation.

    Validates ``output_schema`` against the Outlines subset, grounds the
    extraction in ``content`` with a system + user message, and constrains the
    model to the schema with ``response_format`` (``strict``). The parsed result
    is post-validated against ``output_schema`` (defense in depth) before it is
    returned. ``gpu`` pins the machine profile for deterministic routing;
    ``max_completion_tokens`` caps the output so a large schema does not truncate.
    """
    validate_output_schema(output_schema)

    title = output_schema.get("title")
    schema_name = title if isinstance(title, str) and title else "extraction"
    user = f"{instruction or _DEFAULT_EXTRACT_INSTRUCTION}\n\n<document>\n{content}\n</document>"
    messages = [
        {"role": "system", "content": _EXTRACT_SYSTEM},
        {"role": "user", "content": user},
    ]
    resp = await client.chat_completions(
        model,
        messages,
        response_format={
            "type": "json_schema",
            "json_schema": {"name": schema_name, "schema": output_schema, "strict": True},
        },
        # Deterministic, grounded extraction — no creative sampling.
        temperature=0.0,
        gpu=gpu,
        max_completion_tokens=max_completion_tokens,
    )
    raw = _message_content(resp)
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        msg = f"cluster returned non-JSON content for a json_schema request: {exc}"
        raise StructuredOutputError(msg) from exc
    _validate_against_schema(data, output_schema)
    return data


def _validate_response_format(response_format: Any, *, model: str) -> dict[str, Any] | None:
    """Pre-validate ``response_format``; return the json_schema to post-validate, if any."""
    if not isinstance(response_format, dict):
        msg = "response_format must be a JSON object"
        raise StructuredOutputError(msg)
    rf_type = response_format.get("type")
    if rf_type == "json_schema":
        # ``json_schema`` is caller-supplied (untrusted): guard against a
        # non-dict value so a malformed request raises a clear typed error.
        json_schema = response_format.get("json_schema")
        schema = json_schema.get("schema") if isinstance(json_schema, dict) else None
        if schema is None:
            msg = "response_format.json_schema.schema is required for a json_schema request"
            raise StructuredOutputError(msg)
        validate_output_schema(schema)
        return schema
    if rf_type == "regex":
        # Mirror the gateway ``MAX_REGEX_LEN`` cap (UTF-8 byte length, matching
        # Rust ``str::len``) so an oversized pattern fails fast at the edge.
        regex = response_format.get("regex")
        if isinstance(regex, str) and len(regex.encode("utf-8")) > _MAX_REGEX_LEN:
            byte_len = len(regex.encode("utf-8"))
            msg = f"response_format.regex length {byte_len} bytes exceeds limit ({_MAX_REGEX_LEN})"
            raise StructuredOutputError(msg)
    elif rf_type == "grammar":
        if _is_outlines_backed(model):
            msg = (
                f"Model '{model}' is Outlines-backed and does not support EBNF/grammar "
                "constrained output — only 'json_schema' and 'regex'. Use response_format "
                "{'type': 'json_schema', ...} or {'type': 'regex', ...}, or target an "
                "xgrammar-backed model for EBNF."
            )
            raise StructuredOutputError(msg)
        # Non-Outlines (xgrammar-backed) model: EBNF passes through, but mirror
        # the gateway ``MAX_EBNF_LEN`` source cap (UTF-8 byte length).
        ebnf = response_format.get("grammar")
        if isinstance(ebnf, str) and len(ebnf.encode("utf-8")) > _MAX_EBNF_LEN:
            byte_len = len(ebnf.encode("utf-8"))
            msg = f"response_format.grammar (EBNF) length {byte_len} bytes exceeds limit ({_MAX_EBNF_LEN})"
            raise StructuredOutputError(msg)
    return None


async def generate_structured(
    client: ChatClient,
    *,
    prompt: str,
    response_format: dict[str, Any],
    model: str,
    gpu: str | None = None,
    max_completion_tokens: int | None = None,
) -> str:
    """Generate output constrained by ``response_format`` (schema / regex / grammar).

    Pre-validates the request: a ``json_schema`` is checked against the Outlines
    subset, and an EBNF ``grammar`` request is rejected up front for
    Outlines-backed models. Passes ``response_format`` through unchanged and
    returns the constrained content string (a JSON string for the json modes).
    A ``json_schema`` response is post-validated against the schema (defense in
    depth) before it is returned. ``gpu`` pins the machine profile;
    ``max_completion_tokens`` caps the output so large JSON does not truncate.
    """
    schema = _validate_response_format(response_format, model=model)
    messages = [{"role": "user", "content": prompt}]
    resp = await client.chat_completions(
        model,
        messages,
        response_format=response_format,
        gpu=gpu,
        max_completion_tokens=max_completion_tokens,
    )
    content = _message_content(resp)
    if schema is not None:
        # json_schema mode: confirm the returned string is JSON that conforms,
        # so a grammar-FSM bypass surfaces as a clear error, not silent bad data.
        try:
            parsed = json.loads(content)
        except json.JSONDecodeError as exc:
            msg = f"cluster returned non-JSON content for a json_schema request: {exc}"
            raise StructuredOutputError(msg) from exc
        _validate_against_schema(parsed, schema)
    return content
