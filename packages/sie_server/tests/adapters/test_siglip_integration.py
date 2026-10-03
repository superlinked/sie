"""API-level integration regression for mixed text+image encode batches (#3136).

The Image Search Playground (console/BFF) sends ONE ``/v1/encode`` request
carrying 1 text item + 4 image items against the catalog visual-search model
(``google/siglip2-base-patch16-224``). Before #3138 the image preprocessor
dropped text-only items, the worker future never resolved for the full batch,
and the request hung until the client timed out.

The unit fix is covered at the preprocessor/worker/adapter seams
(``tests/core/test_preprocessor.py``, ``tests/core/test_worker_core.py``,
``tests/adapters/test_siglip.py``). #3136 additionally requires the HTTP API
seam: real request decode -> ``EncodePipeline.run_encode`` -> worker, for both
request shapes:

- the exact console/BFF wire shape: JSON body with base64 ``data`` images
- the native SDK shape: ``SIEClient.encode`` with raw image bytes

Raw ``httpx`` is used deliberately for the console/BFF shape — the regression
is about the wire-level request decode, which the SDK would paper over.

Mark: integration (run with ``mise run test -- -i``). Downloads SigLIP2-base
weights on first run; CPU inference is supported (the adapter forces float32
off-CUDA).
"""

from __future__ import annotations

import base64
import io
import math
import socket
import time
from collections.abc import Generator
from typing import Any

import httpx
import numpy as np
import pytest
from PIL import Image
from sie_sdk import SIEClient

pytestmark = pytest.mark.integration

# The exact catalog model behind the visual-search task in #3136.
MODEL = "google/siglip2-base-patch16-224"
DENSE_DIM = 768

# First request lazy-loads the model (non-blocking 503 MODEL_LOADING while
# weights download + load); this bounds the retry loop, not a single request.
_LOAD_DEADLINE_S = 600.0
# Bounded per-request timeout: the #3136 hang surfaces as a ReadTimeout
# failure here instead of wedging the suite.
_REQUEST_TIMEOUT_S = 120.0
_WARM_TIMEOUT_S = 60.0


def _find_free_port(start: int = 8500, end: int = 8600) -> int:
    for port in range(start, end):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                s.bind(("127.0.0.1", port))
                return port
        except OSError:
            continue
    msg = f"No free port found in range {start}-{end}"
    raise RuntimeError(msg)


@pytest.fixture(scope="module")
def siglip_server(device: str, sie_server_process_factory: type[Any]) -> Generator[str]:
    # The probe socket is closed before the child binds, so a concurrent
    # process can steal the port in between. Retry with a fresh port instead
    # of failing the whole module on that race.
    last_error: Exception | None = None
    for _ in range(3):
        server = sie_server_process_factory(
            port=_find_free_port(),
            models_dir="packages/sie_server/models",
        )
        try:
            server.start(MODEL, device)
            server.wait_ready(timeout_s=600.0)
        except Exception as exc:  # noqa: BLE001 - port stolen or failed boot; retry fresh
            server.stop()
            last_error = exc
            continue
        try:
            yield server.get_url()
        finally:
            server.stop()
        return
    msg = "sie-server failed to start on 3 candidate ports"
    raise RuntimeError(msg) from last_error


def _png_bytes(color: tuple[int, int, int]) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (16, 16), color=color).save(buf, format="PNG")
    return buf.getvalue()


def _mixed_json_items(tag: str) -> list[dict[str, Any]]:
    """The console/BFF shape: 1 text item + 4 base64 image items, one request."""
    items: list[dict[str, Any]] = [{"id": f"{tag}-text-0", "text": f"a red square ({tag})"}]
    for index in range(4):
        png = _png_bytes((255, 4 * index, 0))
        items.append(
            {
                "id": f"{tag}-image-{index + 1}",
                "images": [{"data": base64.b64encode(png).decode("ascii"), "format": "png"}],
            }
        )
    return items


def _post_encode_json(url: str, items: list[dict[str, Any]], *, timeout_s: float) -> httpx.Response:
    """POST the JSON encode request, retrying only the non-blocking lazy-load 503.

    ``/v1/encode`` returns ``503 MODEL_LOADING`` immediately while the model
    loads; each actual inference attempt stays bounded by ``timeout_s`` so a
    reintroduced #3136 hang fails fast instead of wedging the run.
    """
    deadline = time.monotonic() + _LOAD_DEADLINE_S
    while True:
        response = httpx.post(
            f"{url}/v1/encode/{MODEL}",
            json={"items": items},
            headers={"Accept": "application/json"},
            timeout=timeout_s,
        )
        if response.status_code != 503:
            return response
        detail = response.json().get("detail", {})
        if detail.get("code") != "MODEL_LOADING" or time.monotonic() >= deadline:
            return response
        time.sleep(2.0)


class TestMixedTextImageEncodeApi:
    """One request, ``items = [{text}, {image} x4]`` — the #3136 failure shape."""

    def test_mixed_text_image_single_request_completes(self, siglip_server: str) -> None:
        """Console/BFF JSON shape: 200 with 5 aligned embeddings, no hang."""
        items = _mixed_json_items("cold")
        response = _post_encode_json(siglip_server, items, timeout_s=_REQUEST_TIMEOUT_S)

        assert response.status_code == 200, response.text
        data = response.json()
        assert data["model"] == MODEL
        assert len(data["items"]) == len(items)
        # Original order preserved: text item first, then the four images.
        assert [result["id"] for result in data["items"]] == [item["id"] for item in items]
        for result in data["items"]:
            dense = result["dense"]
            assert dense is not None, f"missing dense for {result['id']}"
            assert dense["dims"] == DENSE_DIM
            assert len(dense["values"]) == DENSE_DIM
            assert all(math.isfinite(value) for value in dense["values"])

    def test_mixed_text_image_warm_repeat_completes(self, siglip_server: str) -> None:
        """A warm second mixed request succeeds within a tight bound."""
        # Prime the model (and absorb the lazy load) with the first shape.
        first = _post_encode_json(siglip_server, _mixed_json_items("prime"), timeout_s=_REQUEST_TIMEOUT_S)
        assert first.status_code == 200, first.text

        warm = _post_encode_json(siglip_server, _mixed_json_items("warm"), timeout_s=_WARM_TIMEOUT_S)
        assert warm.status_code == 200, warm.text
        assert len(warm.json()["items"]) == 5

    def test_mixed_text_image_sdk_batch_completes(self, siglip_server: str) -> None:
        """Native SDK shape: raw image bytes through ``SIEClient.encode``."""
        client = SIEClient(siglip_server, timeout_s=float(_LOAD_DEADLINE_S))
        sdk_items: list[dict[str, Any]] = [{"id": "sdk-text-0", "text": "a red square (sdk)"}]
        sdk_items.extend(
            {"id": f"sdk-image-{index + 1}", "images": [_png_bytes((0, 4 * index, 255))]} for index in range(4)
        )

        results = client.encode(MODEL, sdk_items)

        assert isinstance(results, list)
        assert len(results) == len(sdk_items)
        assert [result.get("id") for result in results] == [item["id"] for item in sdk_items]
        for result in results:
            assert isinstance(result["dense"], np.ndarray)
            assert result["dense"].shape == (DENSE_DIM,)
            assert np.isfinite(result["dense"]).all()
