"""Pure helpers for the jobs surface (``client.jobs``) — no transport.

The jobs API is the gateway's batch class. ``client.jobs.submit(...)`` binds
to ``POST /v1/jobs``; this module owns the transport-free pieces so they stay
unit-testable without a gateway:

* :func:`build_job_body` — the ``source → operation → sink / when`` slot
  mapping onto the ``POST /v1/jobs`` body (inline items vs a connector
  ``src``/``sink`` + connection name).
* :func:`connection_name` — derive a connection name from a connector URI.
* :func:`decode_result_item` / :func:`job_chunks` — decode a finished job's
  msgpack chunk refs into per-item results (results are refs, not payloads).
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any, Literal
from urllib.parse import urlsplit

import numpy as np

from sie_sdk._msgpack import unpackb as unpack_msgpack
from sie_sdk.types import JobChunk, JobItemErrorDetail, JobResultItem

if TYPE_CHECKING:
    from numpy.typing import NDArray

# Job states with no further transitions (job lifecycle).
TERMINAL_JOB_STATES = frozenset({"succeeded", "failed", "suspended", "cancelled"})

# Sink sentinels: return the results (default) or write next to the source.
_SINK_RETURN = frozenset({"return", "default"})
_SINK_INPLACE = frozenset({"inplace", "in_place", "in place"})

# Internal push-to-us schemes (OUR Files store) — no org connection to
# name, so no `connection`/`sink_connection` is derived from the URI.
_INTERNAL_SCHEMES = frozenset({"upload"})

# Uniform source-mapping slots (the sink slot is `output_field`).
_FIELD_MAP_KEYS = frozenset({"id_field", "input_field", "carry", "input_type"})
_INPUT_TYPES = frozenset({"text", "document"})

# One canonical path segment across the SDK, gateway, dispatcher, and control
# plane. This is deliberately ASCII-only: connection names cross credential
# resolution and HTTP routing boundaries and must not have decoded aliases.
_CONNECTION_NAME_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}\Z", re.ASCII)
_POSTGRES_SCHEMA_PATTERN = re.compile(r"[A-Za-z_][A-Za-z0-9_$]{0,62}\Z", re.ASCII)

# Public connector actions share the gateway's bounded printable-ASCII header
# contract.  Callers own this value so a logical retry can reuse the same key;
# generating it inside ``submit``/``execute`` would turn a retry into a new
# mutation.
_CONNECTOR_IDEMPOTENCY_KEY_MAX_BYTES = 256


def _norm_item(item: Any, index: int) -> dict[str, Any]:
    """Normalize a job item to the ``/v1/encode`` item contract (``{text}``/``{id,text}``)."""
    if isinstance(item, str):
        return {"text": item}
    if isinstance(item, dict):
        return item
    msg = f"item {index} must be a string or an object, got {type(item).__name__}"
    raise ValueError(msg)


def _is_connector_uri(value: Any) -> bool:
    """A connector source/sink is a ``scheme://…`` URI (inline items are a list)."""
    return isinstance(value, str) and "://" in value


def require_connection_name(name: str) -> str:
    """Return a canonical named-connection path segment or fail before I/O."""
    if _CONNECTION_NAME_PATTERN.fullmatch(name) is None:
        msg = "connection name must be 1-128 ASCII letters, digits, '.', '_', or '-', and start with a letter or digit"
        raise ValueError(msg)
    return name


def require_connection_schema_policy(
    connection_type: str,
    source_schema: str | None,
    sink_schema: str | None,
) -> tuple[str, str] | None:
    """Validate the optional Postgres source/sink namespace pair before I/O."""
    if (source_schema is None) != (sink_schema is None):
        msg = "source_schema and sink_schema must be supplied together"
        raise ValueError(msg)
    if source_schema is None or sink_schema is None:
        return None
    if connection_type != "postgres":
        msg = "source_schema and sink_schema apply only to postgres connections"
        raise ValueError(msg)
    if (
        _POSTGRES_SCHEMA_PATTERN.fullmatch(source_schema) is None
        or _POSTGRES_SCHEMA_PATTERN.fullmatch(sink_schema) is None
    ):
        msg = "source_schema and sink_schema must be canonical Postgres identifiers of at most 63 ASCII bytes"
        raise ValueError(msg)
    return source_schema, sink_schema


def require_connector_idempotency_key(key: str | None) -> str:
    """Return a valid retry-stable connector action idempotency key.

    The gateway requires exactly one ``Idempotency-Key`` header containing
    1-256 printable ASCII bytes.  Keeping the same validation in the SDK fails
    before I/O without weakening the gateway's authoritative check.
    """
    if (
        not isinstance(key, str)
        or not 1 <= len(key) <= _CONNECTOR_IDEMPOTENCY_KEY_MAX_BYTES
        or not all(0x20 <= ord(char) <= 0x7E for char in key)
    ):
        msg = f"connector idempotency_key must contain 1-{_CONNECTOR_IDEMPOTENCY_KEY_MAX_BYTES} printable ASCII bytes"
        raise ValueError(msg)
    return key


def connection_name(uri: str) -> str:
    """The connection an org registered, referenced by the URI authority.

    ``postgres://warehouse?query=…`` → ``warehouse``; ``s3://customer-bucket/in/``
    → ``customer-bucket``. Credentials never appear in the call — the job only
    names the connection; the runner resolves it org-scoped.
    """
    # Read the raw authority rather than a URL parser's normalized netloc:
    # control characters must fail, not disappear into an alias before the
    # canonical path-segment check.
    after_scheme = uri.split("://", 1)[1] if "://" in uri else ""
    name = re.split(r"[/?#]", after_scheme, maxsplit=1)[0]
    if not name:
        msg = f"connector URI {uri!r} names no connection (expected 'scheme://<connection>/…')"
        raise ValueError(msg)
    return require_connection_name(name)


def _is_internal_uri(uri: str) -> bool:
    """True for the internal push-to-us schemes (``upload://`` — OUR store)."""
    return urlsplit(uri).scheme in _INTERNAL_SCHEMES


def _resolve_source(source: Any, connection: str | None) -> dict[str, Any]:
    """Map the ``source`` slot onto jobs-API fields (inline items | connector URI).

    An internal-scheme URI (``upload://<file-id>``) names no org
    connection — the address is OUR Files store — so no ``connection`` rides
    unless explicitly given.
    """
    if isinstance(source, list):
        if not source:
            msg = "inline source has no items"
            raise ValueError(msg)
        return {"items": [_norm_item(item, i) for i, item in enumerate(source)]}
    if _is_connector_uri(source):
        if _is_internal_uri(source):
            return {
                "src": source,
                **({"connection": require_connection_name(connection)} if connection else {}),
            }
        return {
            "src": source,
            "connection": require_connection_name(connection) if connection else connection_name(source),
        }
    if isinstance(source, str) and source.strip():
        # A bare string is one inline text item (the "embed this text" case).
        return {"items": [{"text": source}]}
    msg = "source must be inline items (a list/string) or a connector URI (scheme://<connection>/…)"
    raise ValueError(msg)


def _resolve_sink(sink: Any, *, source_connection: str | None, sink_connection: str | None) -> dict[str, Any]:
    """Map the ``sink`` slot: return (default) | in place | a connector URI."""
    if sink is None or (isinstance(sink, str) and sink.strip().lower() in _SINK_RETURN):
        return {}
    if isinstance(sink, str) and sink.strip().lower() in _SINK_INPLACE:
        return {"sink": "inplace"}
    if _is_connector_uri(sink):
        body: dict[str, Any] = {"sink": sink}
        if _is_internal_uri(sink):
            # Internal scheme: OUR Files store, no connection to name.
            if sink_connection is not None:
                body["sink_connection"] = require_connection_name(sink_connection)
            return body
        resolved = require_connection_name(sink_connection) if sink_connection is not None else connection_name(sink)
        # Thread the sink connection when explicitly overridden or distinct from
        # the source's (the common "index my own store" case reuses the source).
        if sink_connection is not None or resolved != source_connection:
            body["sink_connection"] = resolved
        return body
    msg = f"sink must be 'return', 'inplace', or a connector URI (got {sink!r})"
    raise ValueError(msg)


def _resolve_field_map(field_map: Mapping[str, Any] | None, output_field: str | None) -> dict[str, Any]:
    """Validate + map the uniform slots onto the wire fields.

    ``field_map`` carries the source slots (``id_field``/``input_field``/
    ``carry``/``input_type``); ``output_field`` is the sink slot (≈
    ``response.body``, aliasing PG ``column`` / object-store ``suffix``). Only
    set fields ride the wire (`/v1` additive-only).
    """
    body: dict[str, Any] = {}
    if field_map is not None:
        if not isinstance(field_map, Mapping):
            msg = f"field_map must be a mapping of the uniform slots, got {type(field_map).__name__}"
            raise ValueError(msg)
        unknown = set(field_map) - _FIELD_MAP_KEYS
        if unknown:
            msg = f"unknown field_map key(s) {sorted(unknown)} (known: {sorted(_FIELD_MAP_KEYS)})"
            raise ValueError(msg)
        carry = field_map.get("carry")
        if carry is not None and (
            not isinstance(carry, (list, tuple)) or not all(isinstance(c, str) and c for c in carry)
        ):
            msg = f"field_map.carry must be a list of field names, got {carry!r}"
            raise ValueError(msg)
        input_type = field_map.get("input_type")
        if input_type is not None and input_type not in _INPUT_TYPES:
            msg = f"field_map.input_type must be one of {sorted(_INPUT_TYPES)}, got {input_type!r}"
            raise ValueError(msg)
        mapped = {
            key: field_map[key] for key in ("id_field", "input_field", "input_type") if field_map.get(key) is not None
        }
        if carry:
            mapped["carry"] = list(carry)
        if mapped:
            body["field_map"] = mapped
    if output_field is not None:
        if not isinstance(output_field, str) or not output_field:
            msg = f"output_field must be a non-empty string, got {output_field!r}"
            raise ValueError(msg)
        body["output_field"] = output_field
    return body


def _resolve_when(when: Any) -> dict[str, Any]:
    """Accept the only trigger implemented by the strict public jobs schema."""
    if when is None or (isinstance(when, str) and when.strip().lower() in {"", "now"}):
        return {}
    msg = f"scheduled and watched jobs are not available; omit when or use 'now', got {when!r}"
    raise ValueError(msg)


def build_job_body(
    *,
    source: Any,
    operation: str,
    model: str,
    sink: Any = None,
    connection: str | None = None,
    sink_connection: str | None = None,
    field_map: Mapping[str, Any] | None = None,
    output_field: str | None = None,
    execution: Literal["plan", "run"] | None = None,
    when: Any = None,
    output_types: Sequence[str] | None = None,
    options: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Compose the ``POST /v1/jobs`` body from the source/op/sink/when slots.

    A thin, pure mapping: inline ``items`` or connector ``src``/``sink``
    + connection name, plus an optional trigger. ``connection`` /
    ``sink_connection`` override the names derived from the URIs; ``field_map``
    + ``output_field`` are the uniform mapping slots (connector jobs
    only — per-connector ``id_column``/``text_column``/``column`` params keep
    working as aliases). ``options`` is one opaque job-level operation map,
    applied uniformly to every item (operation matrix: score → ``options.query``,
    extract → ``options.labels`` / ``options.output_schema``, generate →
    sampling such as ``max_new_tokens``); it is forwarded as-is. Only the
    fields that are set ride the wire, so an inline submit is byte-for-byte
    the realtime POC body and the connector body is additive (``/v1``
    additive-only rule). The public contract requires every connector-src
    request to set ``execution``; inline requests must omit it, including for
    callers outside this repository.

    Raises:
        ValueError: If the source/sink/when/field_map/options slots cannot be
            resolved.
    """
    body: dict[str, Any] = {"operation": operation, "model": model}
    source_fields = _resolve_source(source, connection)
    body.update(source_fields)
    sink_fields = _resolve_sink(
        sink,
        source_connection=source_fields.get("connection"),
        sink_connection=sink_connection,
    )
    if "items" in source_fields and (connection is not None or sink_fields or sink_connection is not None):
        msg = "connection/sink/sink_connection apply only to connector-src jobs; inline items return results"
        raise ValueError(msg)
    body.update(sink_fields)
    if "src" in body:
        if execution not in {"plan", "run"}:
            msg = "connector jobs require execution='plan' or execution='run'"
            raise ValueError(msg)
        if execution != "run" and (_is_internal_uri(str(source)) or (isinstance(sink, str) and _is_internal_uri(sink))):
            msg = "upload:// connector jobs are run-only; set execution='run'"
            raise ValueError(msg)
        body["execution"] = execution
    elif execution is not None:
        msg = "execution applies only to connector-src jobs; inline items must omit it"
        raise ValueError(msg)
    mapping_fields = _resolve_field_map(field_map, output_field)
    if mapping_fields and "src" not in body:
        msg = "field_map/output_field apply to connector-src jobs; an inline items job maps nothing"
        raise ValueError(msg)
    body.update(mapping_fields)
    body.update(_resolve_when(when))
    if output_types:
        body["output_types"] = list(output_types)
    if options is not None:
        if not isinstance(options, Mapping):
            msg = f"options must be a mapping (one job-level operation map), got {type(options).__name__}"
            raise ValueError(msg)
        if options:
            body["options"] = dict(options)
    return body


def _to_array(value: Any) -> NDArray[Any] | None:
    if value is None:
        return None
    if isinstance(value, np.ndarray):
        return value
    if isinstance(value, (list, tuple)):
        return np.asarray(value)
    if hasattr(value, "tolist"):
        return np.asarray(value)
    return None


def _dense_info(dense: Any) -> tuple[int | None, NDArray[Any] | None]:
    """Extract (dims, vector) from a decoded dense embedding of unknown shape.

    The vector may be a numpy array (msgpack-numpy) or a plain list, so keys are
    probed with explicit ``is None`` checks — ``a or b`` on an ndarray raises.
    """
    if dense is None:
        return None, None
    if isinstance(dense, dict):
        raw: Any = None
        for candidate in ("values", "vector", "dense"):
            if dense.get(candidate) is not None:
                raw = dense.get(candidate)
                break
        vector = _to_array(raw)
        dims = dense.get("dims")
        if dims is None and vector is not None:
            dims = int(vector.shape[0])
        return dims, vector
    vector = _to_array(dense)
    return (int(vector.shape[0]) if vector is not None else None), vector


def _result_item_error(result: Mapping[str, Any]) -> JobItemErrorDetail | None:
    """Extract a per-item failure from a ``WorkResult`` map (``error``/``error_code``).

    The gateway writes every item's ``WorkResult`` into the chunk ref — including
    failures, each carrying its own ``error`` (free text) and ``error_code`` — so
    a caller can see WHY a specific item failed. Returns ``None`` when the item
    reports no failure signal (a success, or a bare-count chunk error with no
    per-item detail).
    """
    code = result.get("error_code")
    message = result.get("error")
    if code is None and message is None:
        return None
    detail: JobItemErrorDetail = {}
    if code is not None:
        detail["code"] = code
    if message is not None:
        detail["message"] = message
    return detail


def decode_result_item(result: Any) -> JobResultItem:
    """Decode one WorkResult map (from a chunk ref) into a per-item result.

    The chunk's ``result_msgpack`` bytes carry the same wire shape the realtime
    path returns per item; the dense vector decodes to a numpy array (SDK-native,
    like :meth:`SIEClient.encode`). A failed item carries no ``result_msgpack``
    but does carry ``success=False`` plus an ``error``/``error_code`` pair, which
    surfaces as :data:`JobResultItem.error`.
    """
    payload = result.get("result_msgpack") if isinstance(result, dict) else None
    decoded: Any = None
    if isinstance(payload, (bytes, bytearray)):
        try:
            decoded = unpack_msgpack(bytes(payload), numeric_arrays=True)
        except Exception:  # noqa: BLE001 - a malformed payload should not abort retrieval
            decoded = None
    dense = decoded.get("dense") if isinstance(decoded, dict) else None
    dims, vector = _dense_info(dense)
    if isinstance(result, dict):
        # The wire id is ``work_item_id``; older/inline results use ``id``. The
        # bare-count chunk error names no ids, so a per-item id comes only from
        # the item's own WorkResult (never fabricated). Fall back on an explicit
        # ``is None`` check, not ``or``: a falsy-but-valid id (integer ``0`` or an
        # empty string) is a real id and must be preserved, not replaced.
        item_id = result.get("id")
        if item_id is None:
            item_id = result.get("work_item_id")
    else:
        item_id = None
    item: JobResultItem = {
        "id": item_id,
        "success": result.get("success") if isinstance(result, dict) else None,
        "units": result.get("units") if isinstance(result, dict) else None,
        "dims": dims,
        "dense": vector,
    }
    error = _result_item_error(result) if isinstance(result, dict) else None
    if error is not None:
        item["error"] = error
    return item


def job_chunks(job_doc: Mapping[str, Any]) -> list[JobChunk]:
    """The chunk-ref metadata from a job status doc (``output.chunks`` refs)."""
    raw = (job_doc.get("output") or {}).get("chunks") or []
    return [
        {
            "seq": chunk.get("seq"),
            "items": chunk.get("items"),
            "state": chunk.get("state"),
            "ref": chunk.get("ref"),
            "units": chunk.get("units"),
            "credits_charged": chunk.get("credits_charged"),
            "rate_book_version": chunk.get("rate_book_version"),
            "error": chunk.get("error"),
        }
        for chunk in raw
    ]


class MalformedChunkError(Exception):
    """A chunk ref's bytes could not be decoded as a msgpack ``WorkResult`` array.

    Distinct from a chunk that published no ref at all: the bytes exist but are
    garbage (not msgpack, or not a list). That is a DECODE fault, not evidence of
    failed publication or billing, so the caller flags it separately and never
    conflates it with a genuinely-unpublished, already-billed chunk.
    """


def decode_chunk_bytes(raw: bytes) -> list[JobResultItem]:
    """Decode a chunk ref's msgpack ``WorkResult`` array into per-item results.

    Raises:
        MalformedChunkError: If the ref's bytes are not decodable msgpack or do
            not decode to a list. The caller confines this (one bad chunk cannot
            sink the whole ``jobs.results()`` call) and reports it as a decode
            fault, distinct from an unpublished chunk. Per-item decoding stays
            defensive (see :func:`decode_result_item`).
    """
    try:
        results = unpack_msgpack(raw, numeric_arrays=False)
    except Exception as exc:
        # Normalize any decode failure (FormatError, ExtraData, ...) to one signal.
        msg = "chunk ref bytes are not decodable msgpack"
        raise MalformedChunkError(msg) from exc
    if not isinstance(results, list):
        msg = f"chunk ref decoded to {type(results).__name__}, not a WorkResult list"
        raise MalformedChunkError(msg)
    return [decode_result_item(r) for r in results]
