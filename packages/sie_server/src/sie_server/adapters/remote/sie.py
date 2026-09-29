"""Remote adapter for an upstream of kind ``sie``: another SIE deployment.

The profile names an upstream from the server's startup configuration and the
model id the upstream serves. ``encode`` forwards text items to the upstream's
``/v1/encode`` in the SDK's msgpack wire format and returns its dense vectors.
Loading makes no outbound call, holds no weights and uses no accelerator.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar
from urllib.parse import quote

import httpx
import numpy as np
from sie_sdk._msgpack import packb, unpackb
from sie_sdk.client._shared import handle_error, parse_encode_results

from sie_server.adapters._base_adapter import BaseAdapter
from sie_server.adapters._spec import AdapterSpec
from sie_server.adapters._types import ERR_NOT_LOADED
from sie_server.config.upstreams import UpstreamConfigError, UpstreamKind, upstream_for_serving
from sie_server.core.inference_output import EncodeOutput
from sie_server.core.upstream_client import upstream_sync_client

if TYPE_CHECKING:
    from sie_server.types.inputs import Item

_MSGPACK = "application/msgpack"


class RemoteUpstreamError(RuntimeError):
    """The upstream answered, but not with the vectors this profile promises."""


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
        params: dict[str, Any] = {"output_types": ["dense"], "options": {"is_query": is_query}}
        if instruction is not None:
            params["instruction"] = instruction
        body = packb({"items": [{"text": item.text} for item in items], "params": params})
        response = self._client.post(
            f"/v1/encode/{quote(self._upstream_model, safe='/')}",
            content=body,
            headers={"Content-Type": _MSGPACK, "Accept": _MSGPACK},
        )
        if response.status_code >= 400:
            handle_error(response)
        decoded = unpackb(response.content, numeric_arrays=True)
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
        if self._dense_dim is not None and dense.shape[1] != self._dense_dim:
            raise RemoteUpstreamError(
                f"upstream returned {dense.shape[1]}-dimensional vectors, the model declares {self._dense_dim}"
            )
        return EncodeOutput(dense=dense, is_query=is_query, dense_dim=dense.shape[1])
