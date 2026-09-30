"""Remote adapter for an upstream of kind ``sie``: another SIE deployment.

The profile names an upstream from the server's startup configuration and the
model id the upstream serves. ``encode`` forwards text items to the upstream's
``/v1/encode`` in the SDK's msgpack wire format and returns its dense vectors.
Loading makes no outbound call, holds no weights and uses no accelerator.

The upstream is outside this deployment, so its answer is treated as untrusted:
the body is read uncompressed under a size cap and a deadline, and a failure
reaches the caller as fixed text, never as anything the upstream sent. No read
waits longer than ``READ_TIMEOUT_S``, so a call takes at most the connect
timeout plus ``REQUEST_DEADLINE_S`` plus one read timeout.
"""

from __future__ import annotations

import re
import time
from typing import TYPE_CHECKING, Any, ClassVar
from urllib.parse import quote

import httpx
import numpy as np
from sie_sdk._msgpack import packb, unpackb
from sie_sdk.client._shared import parse_encode_results

from sie_server.adapters._base_adapter import BaseAdapter
from sie_server.adapters._spec import AdapterSpec
from sie_server.adapters._types import ERR_NOT_LOADED
from sie_server.config.upstreams import UpstreamConfigError, UpstreamCredentialError, UpstreamKind, upstream_for_serving
from sie_server.core.inference_output import EncodeOutput
from sie_server.core.upstream_client import upstream_sync_client

if TYPE_CHECKING:
    from sie_server.types.inputs import Item

_MSGPACK = "application/msgpack"
REQUEST_DEADLINE_S = 60.0
READ_TIMEOUT_S = 10.0
_CONNECT_TIMEOUT_S = 5.0
_RESPONSE_OVERHEAD_BYTES = 1 << 20
_BYTES_PER_VALUE = 16
_UNKNOWN_DIM_RESPONSE_BYTES = 64 << 20
_ERROR_CODE = re.compile(r"^[A-Z][A-Z0-9_]{0,63}$")


class RemoteUpstreamError(RuntimeError):
    """The upstream call failed or did not return the vectors this profile promises."""


class SieUpstreamAdapter(BaseAdapter):
    """Serve ``encode`` for a remote profile from an SIE upstream."""

    spec: ClassVar[AdapterSpec] = AdapterSpec(inputs=("text",), outputs=("dense",), unload_fields=())

    def __init__(
        self,
        model_name_or_path: str | None = None,
        *,
        upstream: str,
        upstream_model: str,
        dense_dim: int | None = None,
        max_seq_length: int | None = None,
        compute_precision: str | None = None,
        **kwargs: Any,
    ) -> None:
        _ = (model_name_or_path, max_seq_length, compute_precision, kwargs)
        self._upstream_name = upstream
        self._upstream_model = upstream_model
        self._dense_dim = dense_dim
        self._client: httpx.Client | None = None
        self._device: str | None = None

    @property
    def upstream_name(self) -> str:
        return self._upstream_name

    def load(self, device: str) -> None:
        upstream = upstream_for_serving(self._upstream_name)
        if upstream.kind is not UpstreamKind.SIE:
            raise UpstreamConfigError(f"upstream {self._upstream_name!r} is not of kind 'sie'")
        self._client = upstream_sync_client(upstream)
        self._device = device

    def unload(self) -> None:
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
        _ = (output_types, prepared_items, options)
        if self._client is None:
            raise RuntimeError(ERR_NOT_LOADED)
        if any(not isinstance(item.text, str) for item in items):
            raise ValueError("a remote SIE profile encodes text items only")
        params: dict[str, Any] = {"output_types": ["dense"], "options": {"is_query": is_query}}
        if instruction is not None:
            params["instruction"] = instruction
        request = self._client.build_request(
            "POST",
            f"/v1/encode/{quote(self._upstream_model, safe='/:')}",
            content=packb({"items": [{"text": item.text} for item in items], "params": params}),
            headers={"Content-Type": _MSGPACK, "Accept": _MSGPACK, "Accept-Encoding": "identity"},
            timeout=httpx.Timeout(READ_TIMEOUT_S, connect=_CONNECT_TIMEOUT_S),
        )
        encode_prefix = self._client.base_url.raw_path.rstrip(b"/") + b"/v1/encode/"
        if not request.url.raw_path.startswith(encode_prefix):
            raise RemoteUpstreamError("the upstream model id does not form an encode path")
        body = self._send(self._client, request, max_bytes=self._max_response_bytes(len(items)))
        try:
            decoded = unpackb(body, numeric_arrays=True)
            wire_items = decoded.get("items") if isinstance(decoded, dict) else None
            if not isinstance(wire_items, list) or len(wire_items) != len(items):
                raise RemoteUpstreamError("upstream returned a different number of results than items sent")
            vectors: list[np.ndarray] = []
            for result in parse_encode_results(wire_items):
                vector = result.get("dense")
                if vector is None or vector.ndim != 1:
                    raise RemoteUpstreamError("upstream returned an item without a dense vector")
                vectors.append(vector)
            dense = np.stack(vectors).astype(np.float32, copy=False)
        except RemoteUpstreamError:
            raise
        except Exception:  # noqa: BLE001 - untrusted bytes; the reason must not reach the caller
            raise RemoteUpstreamError("upstream returned a body that is not an encode response") from None
        if self._dense_dim is not None and dense.shape[1] != self._dense_dim:
            raise RemoteUpstreamError(
                f"upstream returned {dense.shape[1]}-dimensional vectors, the model declares {self._dense_dim}"
            )
        if not np.isfinite(dense).all():
            raise RemoteUpstreamError("upstream returned non-finite values")
        return EncodeOutput(dense=dense, is_query=is_query, dense_dim=dense.shape[1])

    def _max_response_bytes(self, item_count: int) -> int:
        if self._dense_dim is None:
            return _UNKNOWN_DIM_RESPONSE_BYTES
        return _RESPONSE_OVERHEAD_BYTES + item_count * self._dense_dim * _BYTES_PER_VALUE

    @staticmethod
    def _send(client: httpx.Client, request: httpx.Request, *, max_bytes: int) -> bytes:
        deadline = time.monotonic() + REQUEST_DEADLINE_S
        try:
            response = client.send(request, stream=True)
        except UpstreamCredentialError:
            raise RemoteUpstreamError("the upstream credential is unavailable") from None
        except httpx.HTTPError as exc:
            raise RemoteUpstreamError(f"upstream request failed ({type(exc).__name__})") from None
        try:
            if response.status_code >= 400:
                code = response.headers.get("x-sie-error-code", "")
                suffix = f" {code}" if _ERROR_CODE.fullmatch(code) else ""
                raise RemoteUpstreamError(f"upstream answered {response.status_code}{suffix}")
            if response.headers.get("content-encoding", "identity").strip().lower() not in {"", "identity"}:
                raise RemoteUpstreamError("upstream sent a compressed body, which is refused")
            chunks: list[bytes] = []
            size = 0
            for chunk in response.iter_raw():
                size += len(chunk)
                if size > max_bytes:
                    raise RemoteUpstreamError("upstream response exceeds the size limit")
                if time.monotonic() > deadline:
                    raise RemoteUpstreamError("upstream response exceeded the deadline")
                chunks.append(chunk)
            return b"".join(chunks)
        except httpx.HTTPError as exc:
            raise RemoteUpstreamError(f"upstream request failed ({type(exc).__name__})") from None
        finally:
            response.close()
