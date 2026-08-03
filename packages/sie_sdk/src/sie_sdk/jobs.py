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

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any
from urllib.parse import urlsplit

import msgpack
import msgpack_numpy
import numpy as np

from sie_sdk.types import JobChunk, JobResultItem

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

# msgpack + the numpy codec are hard deps of the SDK, so always importable here.
_RESULT_OBJECT_HOOK = msgpack_numpy.decode


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


def connection_name(uri: str) -> str:
    """The connection an org registered, referenced by the URI authority.

    ``postgres://warehouse?query=…`` → ``warehouse``; ``s3://customer-bucket/in/``
    → ``customer-bucket``. Credentials never appear in the call — the job only
    names the connection; the runner resolves it org-scoped.
    """
    parts = urlsplit(uri)
    name = parts.netloc or parts.path.lstrip("/").split("/", 1)[0]
    if not name:
        msg = f"connector URI {uri!r} names no connection (expected 'scheme://<connection>/…')"
        raise ValueError(msg)
    return name


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
            return {"src": source, **({"connection": connection} if connection else {})}
        return {"src": source, "connection": connection or connection_name(source)}
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
                body["sink_connection"] = sink_connection
            return body
        resolved = sink_connection if sink_connection is not None else connection_name(sink)
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
    """Map the ``when`` trigger: now (default) | schedule(cron) | watch(source)."""
    if when is None or not isinstance(when, str) or when.strip().lower() in {"", "now"}:
        return {}
    text = when.strip()
    if text.lower().startswith("schedule:"):
        return {"when": "schedule", "schedule": text.split(":", 1)[1].strip()}
    if text.lower().startswith("watch:"):
        return {"when": "watch", "watch": text.split(":", 1)[1].strip()}
    if text.lower() == "schedule":
        msg = "schedule trigger needs a cron expr: when='schedule:<cron>'"
        raise ValueError(msg)
    # A bare cron expression (5 whitespace-separated fields) is a schedule.
    if len(text.split()) == 5:
        return {"when": "schedule", "schedule": text}
    msg = f"unrecognized when {when!r}: use 'now', 'schedule:<cron>', or 'watch:<source>'"
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
    working as aliases). ``options`` is the opaque per-item options map plus
    the op inputs (operation matrix: score → ``options.query``,
    extract → ``options.labels`` / ``options.output_schema``, generate →
    sampling such as ``max_new_tokens``); it is forwarded as-is. Only the
    fields that are set ride the wire, so an inline submit is byte-for-byte
    the realtime POC body and the connector body is additive (``/v1``
    additive-only rule).

    Raises:
        ValueError: If the source/sink/when/field_map/options slots cannot be
            resolved.
    """
    body: dict[str, Any] = {"operation": operation, "model": model}
    source_fields = _resolve_source(source, connection)
    body.update(source_fields)
    body.update(_resolve_sink(sink, source_connection=source_fields.get("connection"), sink_connection=sink_connection))
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
            msg = f"options must be a mapping (per-item options + op inputs), got {type(options).__name__}"
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


def decode_result_item(result: Any) -> JobResultItem:
    """Decode one WorkResult map (from a chunk ref) into a per-item result.

    The chunk's ``result_msgpack`` bytes carry the same wire shape the realtime
    path returns per item; the dense vector decodes to a numpy array (SDK-native,
    like :meth:`SIEClient.encode`).
    """
    payload = result.get("result_msgpack") if isinstance(result, dict) else None
    decoded: Any = None
    if isinstance(payload, (bytes, bytearray)):
        try:
            decoded = msgpack.unpackb(payload, raw=False, object_hook=_RESULT_OBJECT_HOOK)
        except Exception:  # noqa: BLE001 - a malformed payload should not abort retrieval
            decoded = None
    dense = decoded.get("dense") if isinstance(decoded, dict) else None
    dims, vector = _dense_info(dense)
    item: JobResultItem = {
        "id": result.get("id") if isinstance(result, dict) else None,
        "success": result.get("success") if isinstance(result, dict) else None,
        "units": result.get("units") if isinstance(result, dict) else None,
        "dims": dims,
        "dense": vector,
    }
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


def decode_chunk_bytes(raw: bytes) -> list[JobResultItem]:
    """Decode a chunk ref's msgpack ``WorkResult`` array into per-item results."""
    results = msgpack.unpackb(raw, raw=False)
    if not isinstance(results, list):
        return []
    return [decode_result_item(r) for r in results]
