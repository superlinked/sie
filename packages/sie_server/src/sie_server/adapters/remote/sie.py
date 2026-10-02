"""Remote adapter for an upstream of kind ``sie``: another SIE deployment.

The profile names an upstream from the server's startup configuration and the
model id the upstream serves. ``encode``, ``score`` and ``extract`` forward text
and image items to the upstream's native routes in the SDK's msgpack wire
format, and every output shape comes back: dense, sparse and multivector
vectors, scores, and extractions. Loading makes no outbound call, holds no
weights and uses no accelerator.

Usage is the upstream's own count. One upstream request is sent for each item
encoded or extracted from, and one for each API request scored, as
:mod:`sie_server.adapters.remote._batching` describes, so each count is exact
for the work it is attributed to.

The upstream is outside this deployment, so its answer is treated as untrusted
by the rules in :mod:`sie_server.adapters.remote._http`: the body is read
uncompressed under a size cap and a deadline, every field is validated before
it is used, and a failure reaches the caller as fixed text, never as anything
the upstream sent. A failure the same request may survive later, such as an
upstream that is still loading the model, raises
:class:`~sie_server.adapters.errors.UpstreamUnavailableError`. No read waits
longer than ``READ_TIMEOUT_S``, so a call takes at most the connect timeout
plus ``REQUEST_DEADLINE_S`` plus one read timeout.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, ClassVar
from urllib.parse import quote

import httpx
import numpy as np
from sie_sdk._msgpack import packb, unpackb

from sie_server.adapters._base_adapter import BaseAdapter
from sie_server.adapters._spec import AdapterSpec
from sie_server.adapters._types import ERR_NOT_LOADED
from sie_server.adapters.remote._batching import (
    RequestScores,
    UpstreamUsage,
    call_each,
    requests_in_flight,
    score_each_request,
)
from sie_server.adapters.remote._http import RemoteUpstreamError, send_bounded
from sie_server.adapters.remote._limits import upstream_limiter
from sie_server.config.upstreams import UpstreamConfigError, UpstreamKind, upstream_for_serving
from sie_server.core.inference_output import EncodeOutput, ExtractItemError, ExtractOutput, ScoreOutput, SparseVector
from sie_server.core.postprocessor_registry import POSTPROCESSOR_OPTION_KEYS
from sie_server.core.upstream_client import upstream_sync_client
from sie_server.types.inputs import InvalidInputError, media_bytes
from sie_server.types.responses import Classification, DetectedObject, Entity, ErrorCode, Relation

if TYPE_CHECKING:
    from sie_server.types.inputs import Item

_MSGPACK = "application/msgpack"
REQUEST_DEADLINE_S = 60.0
READ_TIMEOUT_S = 10.0
_CONNECT_TIMEOUT_S = 5.0
_RESPONSE_OVERHEAD_BYTES = 1 << 20
_BYTES_PER_VALUE = 16
_SCORE_ENTRY_BYTES = 256
_LARGEST_RESPONSE_BYTES = 64 << 20
_MAX_REPORTED_COUNT = 1 << 32
_MAX_DATA_DEPTH = 64
_ENCODE_OUTPUTS = ("dense", "sparse", "multivector")
_PRIMITIVES = {"encode": "an encode", "score": "a score", "extract": "an extract"}
_LOCAL_OPTION_KEYS = POSTPROCESSOR_OPTION_KEYS | {
    "profile",
    "lora",
    "lora_id",
    "output_types",
    "is_query",
    "instruction",
}
_ITEM_ERRORS = {
    ErrorCode.INVALID_INPUT.value: "the upstream refused this item",
    ErrorCode.INPUT_TOO_LONG.value: "the upstream refused this item as too long",
}
_ITEM_FAILED = "the upstream could not extract from this item"


@dataclass(frozen=True, slots=True)
class _Encoded:
    dense: np.ndarray | None
    sparse: SparseVector | None
    multivector: np.ndarray | None
    usage: UpstreamUsage | None


@dataclass(frozen=True, slots=True)
class _Extracted:
    entities: list[Entity]
    relations: list[Relation]
    classifications: list[Classification]
    objects: list[DetectedObject]
    data: dict[str, Any]
    error: ExtractItemError | None
    usage: UpstreamUsage | None


class SieUpstreamAdapter(BaseAdapter):
    """Serve ``encode``, ``score`` and ``extract`` for a remote profile from an SIE upstream."""

    spec: ClassVar[AdapterSpec] = AdapterSpec(
        inputs=("text", "image"),
        outputs=("dense", "sparse", "multivector", "score", "json"),
        unload_fields=(),
    )

    def __init__(
        self,
        model_name_or_path: str | None = None,
        *,
        upstream: str,
        upstream_model: str,
        dense_dim: int | None = None,
        sparse_dim: int | None = None,
        multivector_dim: int | None = None,
        max_seq_length: int | None = None,
        compute_precision: str | None = None,
        **kwargs: Any,
    ) -> None:
        _ = (model_name_or_path, max_seq_length, compute_precision, kwargs)
        self._upstream_name = upstream
        self._upstream_model = upstream_model
        self._dense_dim = dense_dim
        self._sparse_dim = sparse_dim
        self._multivector_dim = multivector_dim
        self._client: httpx.Client | None = None
        self._executor: ThreadPoolExecutor | None = None
        self._device: str | None = None

    @property
    def upstream_name(self) -> str:
        return self._upstream_name

    def load(self, device: str) -> None:
        upstream = upstream_for_serving(self._upstream_name)
        if upstream.kind is not UpstreamKind.SIE:
            raise UpstreamConfigError(f"upstream {self._upstream_name!r} is not of kind 'sie'")
        self._client = upstream_sync_client(upstream)
        self._executor = ThreadPoolExecutor(
            max_workers=requests_in_flight(upstream), thread_name_prefix=f"sie-upstream-{self._upstream_name}"
        )
        self._device = device

    def unload(self) -> None:
        executor, self._executor = self._executor, None
        if executor is not None:
            executor.shutdown(wait=False, cancel_futures=True)
        client, self._client = self._client, None
        if client is not None:
            client.close()
        super().unload()

    def encode(
        self,
        items: list[Item],
        output_types: list[str],
        *,
        instruction: str | None = None,
        is_query: bool = False,
        prepared_items: list[Any] | None = None,
        options: dict[str, Any] | None = None,
    ) -> EncodeOutput:
        _ = prepared_items
        executor = self._loaded_executor()
        requested = [output for output in _ENCODE_OUTPUTS if output in output_types] or ["dense"]
        params: dict[str, Any] = {
            "output_types": requested,
            "output_dtype": "float32",
            "options": {**_forwarded_options(options), "is_query": is_query},
        }
        if instruction is not None:
            params["instruction"] = instruction
        wire_items = [_wire_item(item) for item in items]
        with upstream_limiter(self._upstream_name).batch(
            len(wire_items), concurrency=requests_in_flight(upstream_for_serving(self._upstream_name))
        ):
            encoded = call_each(executor, lambda wire_item: self._encode_one(wire_item, params, requested), wire_items)
        extra: dict[str, Any] = {}
        token_counts = _reported([None if answer.usage is None else answer.usage.input_tokens for answer in encoded])
        if token_counts is not None:
            extra["input_token_counts"] = token_counts
        image_counts = _reported([None if answer.usage is None else answer.usage.images for answer in encoded])
        if image_counts is not None:
            extra["input_image_counts"] = image_counts
        return EncodeOutput(
            dense=_stacked([answer.dense for answer in encoded]) if "dense" in requested else None,
            sparse=[_present(answer.sparse) for answer in encoded] if "sparse" in requested else None,
            multivector=_same_token_dims([answer.multivector for answer in encoded])
            if "multivector" in requested
            else None,
            batch_size=len(items),
            is_query=is_query,
            extra=extra,
        )

    def score_pairs(
        self,
        queries: list[Item],
        docs: list[Item],
        *,
        instruction: str | None = None,
        options: dict[str, Any] | None = None,
    ) -> ScoreOutput:
        executor = self._loaded_executor()
        forwarded = _forwarded_options(options)
        wire = {id(item): _wire_item(item) for item in (*queries, *docs)}

        def score_request(query: Item, request_docs: list[Item]) -> RequestScores:
            payload: dict[str, Any] = {
                "query": wire[id(query)],
                "items": [{**wire[id(doc)], "id": str(index)} for index, doc in enumerate(request_docs)],
            }
            if instruction is not None:
                payload["instruction"] = instruction
            if forwarded:
                payload["options"] = forwarded
            decoded = self._post(
                "score", payload, max_bytes=_RESPONSE_OVERHEAD_BYTES + len(request_docs) * _SCORE_ENTRY_BYTES
            )
            return RequestScores(scores=_scores(decoded, len(request_docs)), usage=_usage(decoded))

        return score_each_request(executor, queries, docs, score_request, limiter=upstream_limiter(self._upstream_name))

    def extract(
        self,
        items: list[Item],
        *,
        labels: list[str] | None = None,
        output_schema: dict[str, Any] | None = None,
        instruction: str | None = None,
        options: dict[str, Any] | None = None,
        prepared_items: list[Any] | None = None,
    ) -> ExtractOutput:
        _ = prepared_items
        executor = self._loaded_executor()
        params: dict[str, Any] = {}
        if labels is not None:
            params["labels"] = labels
        if output_schema is not None:
            params["output_schema"] = output_schema
        if instruction is not None:
            params["instruction"] = instruction
        if forwarded := _forwarded_options(options):
            params["options"] = forwarded
        wire_items = [_wire_item(item) for item in items]
        with upstream_limiter(self._upstream_name).batch(
            len(wire_items), concurrency=requests_in_flight(upstream_for_serving(self._upstream_name))
        ):
            extracted = call_each(executor, lambda wire_item: self._extract_one(wire_item, params), wire_items)
        errors = [answer.error for answer in extracted]
        return ExtractOutput(
            entities=[answer.entities for answer in extracted],
            relations=[answer.relations for answer in extracted],
            classifications=[answer.classifications for answer in extracted],
            objects=[answer.objects for answer in extracted],
            data=[answer.data for answer in extracted],
            errors=errors if any(error is not None for error in errors) else None,
            batch_size=len(items),
            input_token_counts=_reported(
                [None if answer.usage is None else answer.usage.input_tokens for answer in extracted]
            ),
        )

    def _loaded_executor(self) -> ThreadPoolExecutor:
        if self._client is None or self._executor is None:
            raise RuntimeError(ERR_NOT_LOADED)
        return self._executor

    def _encode_one(self, wire_item: dict[str, Any], params: dict[str, Any], requested: list[str]) -> _Encoded:
        decoded = self._post(
            "encode", {"items": [wire_item], "params": params}, max_bytes=self._encode_response_bytes(requested)
        )
        result = _single_result(decoded)
        return _Encoded(
            dense=self._dense(result) if "dense" in requested else None,
            sparse=self._sparse(result) if "sparse" in requested else None,
            multivector=self._multivector(result) if "multivector" in requested else None,
            usage=_usage(decoded),
        )

    def _extract_one(self, wire_item: dict[str, Any], params: dict[str, Any]) -> _Extracted:
        payload: dict[str, Any] = {"items": [wire_item]}
        if params:
            payload["params"] = params
        decoded = self._post("extract", payload, max_bytes=_LARGEST_RESPONSE_BYTES)
        result = _single_result(decoded)
        return _Extracted(
            entities=[_entity(value) for value in _listed(result, "entities")],
            relations=[_relation(value) for value in _listed(result, "relations")],
            classifications=[_classification(value) for value in _listed(result, "classifications")],
            objects=[_object(value) for value in _listed(result, "objects")],
            data=_data(result.get("data")),
            error=_item_error(result.get("error")),
            usage=_usage(decoded),
        )

    def _post(self, primitive: str, payload: dict[str, Any], *, max_bytes: int) -> dict[str, Any]:
        client = self._client
        if client is None:
            raise RuntimeError(ERR_NOT_LOADED)
        request = client.build_request(
            "POST",
            f"/v1/{primitive}/{quote(self._upstream_model, safe='/:')}",
            content=packb(payload),
            headers={"Content-Type": _MSGPACK, "Accept": _MSGPACK, "Accept-Encoding": "identity"},
            timeout=httpx.Timeout(READ_TIMEOUT_S, connect=_CONNECT_TIMEOUT_S),
        )
        route_prefix = client.base_url.raw_path.rstrip(b"/") + f"/v1/{primitive}/".encode()
        if not request.url.raw_path.startswith(route_prefix):
            raise RemoteUpstreamError(f"the upstream model id does not form {_PRIMITIVES[primitive]} path")
        body = send_bounded(
            client, request, upstream=self._upstream_name, max_bytes=max_bytes, deadline_s=REQUEST_DEADLINE_S
        )
        try:
            decoded = unpackb(body, numeric_arrays=True)
        except Exception:  # noqa: BLE001 - untrusted bytes; the reason must not reach the caller
            decoded = None
        if not isinstance(decoded, dict):
            raise RemoteUpstreamError(f"upstream returned a body that is not {_PRIMITIVES[primitive]} response")
        return decoded

    def _encode_response_bytes(self, requested: list[str]) -> int:
        dims = {"dense": self._dense_dim, "sparse": None if self._sparse_dim is None else 2 * self._sparse_dim}
        budget = _RESPONSE_OVERHEAD_BYTES
        for output in requested:
            values = dims.get(output)
            if values is None:
                return _LARGEST_RESPONSE_BYTES
            budget += values * _BYTES_PER_VALUE
        return budget

    def _dense(self, result: dict[str, Any]) -> np.ndarray:
        vector = _values(result.get("dense"))
        if vector is None or vector.ndim != 1:
            raise RemoteUpstreamError("upstream returned an item without a dense vector")
        if self._dense_dim is not None and vector.shape[0] != self._dense_dim:
            raise RemoteUpstreamError(
                f"upstream returned {vector.shape[0]}-dimensional vectors, the model declares {self._dense_dim}"
            )
        return _finite(vector)

    def _sparse(self, result: dict[str, Any]) -> SparseVector:
        sparse = result.get("sparse")
        indices = sparse.get("indices") if isinstance(sparse, Mapping) else None
        values = _values(sparse)
        if (
            not isinstance(indices, np.ndarray)
            or values is None
            or indices.ndim != 1
            or values.ndim != 1
            or indices.shape != values.shape
            or not np.issubdtype(indices.dtype, np.integer)
        ):
            raise RemoteUpstreamError("upstream returned an item without a sparse vector")
        limit = self._sparse_dim if self._sparse_dim is not None else 1 << 31
        if indices.size and (int(indices.min()) < 0 or int(indices.max()) >= limit):
            raise RemoteUpstreamError("upstream returned sparse indices outside the model's dimensions")
        return SparseVector(indices=indices.astype(np.int32), values=_finite(values))

    def _multivector(self, result: dict[str, Any]) -> np.ndarray:
        vectors = _values(result.get("multivector"))
        if vectors is None or vectors.ndim != 2:
            raise RemoteUpstreamError("upstream returned an item without a multivector")
        if self._multivector_dim is not None and vectors.shape[1] != self._multivector_dim:
            raise RemoteUpstreamError(
                f"upstream returned {vectors.shape[1]}-dimensional token vectors, "
                f"the model declares {self._multivector_dim}"
            )
        return _finite(vectors)


def _forwarded_options(options: dict[str, Any] | None) -> dict[str, Any]:
    """The runtime options the upstream applies; the ones this server consumes stay here."""
    return {key: value for key, value in (options or {}).items() if key not in _LOCAL_OPTION_KEYS}


def _wire_item(item: Item) -> dict[str, Any]:
    """``item`` in the SDK's wire format. Only text and images are forwarded."""
    if item.audio is not None or item.video is not None or item.document is not None:
        raise InvalidInputError("a remote SIE profile forwards text and image inputs only")
    if item.text is not None and not isinstance(item.text, str):
        raise InvalidInputError("item text must be a string")
    wire: dict[str, Any] = {}
    if item.text is not None:
        wire["text"] = item.text
    if item.images:
        wire["images"] = [_wire_image(image) for image in item.images]
    return wire


def _wire_image(image: Any) -> dict[str, Any]:
    data = media_bytes(image, kind="image")
    image_format = image.get("format") if isinstance(image, Mapping) else None
    return {"data": data, "format": image_format if isinstance(image_format, str) else None}


def _single_result(decoded: dict[str, Any]) -> dict[str, Any]:
    results = decoded.get("items")
    if not isinstance(results, list) or len(results) != 1 or not isinstance(results[0], dict):
        raise RemoteUpstreamError("upstream returned a different number of results than items sent")
    return results[0]


def _values(field: Any) -> np.ndarray | None:
    values = field.get("values") if isinstance(field, Mapping) else None
    if not isinstance(values, np.ndarray) or not np.issubdtype(values.dtype, np.floating):
        return None
    return values


def _finite(values: np.ndarray) -> np.ndarray:
    if not np.isfinite(values).all():
        raise RemoteUpstreamError("upstream returned non-finite values")
    return values.astype(np.float32, copy=False)


def _stacked(vectors: list[np.ndarray | None]) -> np.ndarray:
    present = [_present(vector) for vector in vectors]
    if len({vector.shape for vector in present}) > 1:
        raise RemoteUpstreamError("upstream returned vectors of different dimensions")
    return np.stack(present)


def _same_token_dims(multivectors: list[np.ndarray | None]) -> list[np.ndarray]:
    present = [_present(vectors) for vectors in multivectors]
    if len({vectors.shape[1] for vectors in present}) > 1:
        raise RemoteUpstreamError("upstream returned token vectors of different dimensions")
    return present


def _present(value: Any) -> Any:
    if value is None:
        raise RemoteUpstreamError("upstream returned an item without a requested output")
    return value


def _reported(counts: list[int | None]) -> list[int] | None:
    """Per-item counts when every item's upstream request reported one, else ``None``."""
    reported = [count for count in counts if count is not None]
    return reported if len(reported) == len(counts) else None


def _usage(decoded: dict[str, Any]) -> UpstreamUsage | None:
    """The usage the upstream reported for its whole request, or ``None`` when it reported none."""
    usage = decoded.get("usage")
    if usage is None:
        return None
    if not isinstance(usage, dict) or "input_tokens" not in usage:
        raise RemoteUpstreamError("upstream reported malformed usage")
    input_tokens = _count(usage["input_tokens"])
    images = None if usage.get("images") is None else _count(usage["images"])
    content_tokens = None
    details = usage.get("input_tokens_details")
    if details is not None:
        if not isinstance(details, dict) or "content_tokens" not in details:
            raise RemoteUpstreamError("upstream reported malformed usage")
        content_tokens = _count(details["content_tokens"])
        if content_tokens > input_tokens:
            raise RemoteUpstreamError("upstream reported malformed usage")
    return UpstreamUsage(input_tokens=input_tokens, images=images, content_tokens=content_tokens)


def _count(value: Any) -> int:
    if isinstance(value, bool | np.bool_) or not isinstance(value, int | np.integer):
        raise RemoteUpstreamError("upstream reported malformed usage")
    count = int(value)
    if not 0 <= count < _MAX_REPORTED_COUNT:
        raise RemoteUpstreamError("upstream reported malformed usage")
    return count


def _scores(decoded: dict[str, Any], count: int) -> list[float]:
    """The upstream's score for each document sent, in the order sent, matched by positional id."""
    entries = decoded.get("scores")
    if not isinstance(entries, list) or len(entries) != count:
        raise RemoteUpstreamError("upstream returned a different number of scores than documents sent")
    positions = {str(position): position for position in range(count)}
    scores: list[float | None] = [None] * count
    for entry in entries:
        item_id = entry.get("item_id") if isinstance(entry, dict) else None
        position = positions.get(item_id) if isinstance(item_id, str) else None
        if position is None or scores[position] is not None:
            raise RemoteUpstreamError("upstream returned scores that do not match the documents sent")
        scores[position] = _number(entry.get("score"), "upstream returned a score that is not a finite number")
    return [float(score or 0.0) for score in scores]


def _number(value: Any, problem: str) -> float:
    if isinstance(value, bool | np.bool_) or not isinstance(value, int | float | np.integer | np.floating):
        raise RemoteUpstreamError(problem)
    number = float(value)
    if not math.isfinite(number):
        raise RemoteUpstreamError(problem)
    return number


def _listed(result: dict[str, Any], key: str) -> list[Any]:
    value = result.get(key)
    if value is None:
        return []
    if not isinstance(value, list):
        raise _malformed_extraction()
    return value


def _malformed_extraction() -> RemoteUpstreamError:
    return RemoteUpstreamError("upstream returned a malformed extraction")


def _text(value: Mapping[str, Any], key: str) -> str:
    text = value.get(key)
    if not isinstance(text, str):
        raise _malformed_extraction()
    return text


def _score(value: Mapping[str, Any]) -> float:
    return _number(value.get("score"), "upstream returned a malformed extraction")


def _offset(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool | np.bool_) or not isinstance(value, int | np.integer) or int(value) < 0:
        raise _malformed_extraction()
    return int(value)


def _bbox(value: Any) -> list[int]:
    if not isinstance(value, list) or len(value) != 4:
        raise _malformed_extraction()
    if any(isinstance(side, bool | np.bool_) or not isinstance(side, int | np.integer) for side in value):
        raise _malformed_extraction()
    return [int(side) for side in value]


def _entity(value: Any) -> Entity:
    if not isinstance(value, dict):
        raise _malformed_extraction()
    bbox = value.get("bbox")
    return Entity(
        text=_text(value, "text"),
        label=_text(value, "label"),
        score=_score(value),
        start=_offset(value.get("start")),
        end=_offset(value.get("end")),
        bbox=None if bbox is None else _bbox(bbox),
    )


def _relation(value: Any) -> Relation:
    if not isinstance(value, dict):
        raise _malformed_extraction()
    return Relation(
        head=_text(value, "head"), tail=_text(value, "tail"), relation=_text(value, "relation"), score=_score(value)
    )


def _classification(value: Any) -> Classification:
    if not isinstance(value, dict):
        raise _malformed_extraction()
    return Classification(label=_text(value, "label"), score=_score(value))


def _object(value: Any) -> DetectedObject:
    if not isinstance(value, dict):
        raise _malformed_extraction()
    return DetectedObject(label=_text(value, "label"), score=_score(value), bbox=_bbox(value.get("bbox")))


def _data(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise _malformed_extraction()
    return _plain(value, 0)


def _plain(value: Any, depth: int) -> Any:
    """``value`` as plain JSON data, or raise when it is not JSON data."""
    if depth > _MAX_DATA_DEPTH:
        raise _malformed_extraction()
    if value is None or isinstance(value, str | bool):
        return value
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, int | np.integer):
        return int(value)
    if isinstance(value, float | np.floating):
        return _number(value, "upstream returned a malformed extraction")
    if isinstance(value, np.ndarray):
        return _plain(value.tolist(), depth)
    if isinstance(value, list):
        return [_plain(element, depth + 1) for element in value]
    if isinstance(value, dict) and all(isinstance(key, str) for key in value):
        return {key: _plain(element, depth + 1) for key, element in value.items()}
    raise _malformed_extraction()


def _item_error(value: Any) -> ExtractItemError | None:
    """An item the upstream could not extract from, under a code this server knows and fixed text."""
    if value is None:
        return None
    code = value.get("code") if isinstance(value, dict) else None
    if isinstance(code, str) and code in _ITEM_ERRORS:
        return ExtractItemError(code=code, message=_ITEM_ERRORS[code])
    return ExtractItemError(code=ErrorCode.INFERENCE_ERROR.value, message=_ITEM_FAILED)
