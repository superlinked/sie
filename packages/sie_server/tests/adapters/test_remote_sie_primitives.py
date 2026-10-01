"""Every encode output, score and extract through an SIE upstream, with the upstream's usage.

The upstream is a real SIE app on loopback serving a fake model that declares
every encode output, score and extract, so requests and answers cross the
actual wire format. The adapter-level cases answer from a mock transport to
reach answers a real upstream never sends.
"""

from __future__ import annotations

import asyncio
import base64
import io
import json
import threading
import time
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import httpx
import numpy as np
import pytest
import yaml
from fastapi.testclient import TestClient
from PIL import Image
from sie_sdk import SIEClient
from sie_sdk._msgpack import packb, unpackb
from sie_server.adapters.remote import _http as remote_http
from sie_server.adapters.remote import sie as remote_sie
from sie_server.adapters.remote._batching import MAX_REQUESTS_IN_FLIGHT
from sie_server.config.model import ModelConfig
from sie_server.config.upstreams import Upstream, install_upstreams
from sie_server.core.loader import load_adapter
from sie_server.core.registry import ModelRegistry
from sie_server.core.upstream_client import upstream_sync_client
from sie_server.ipc_types import (
    EncodeBatchItem,
    ExtractBatchItem,
    ProcessEncodeBatchRequest,
    ProcessExtractBatchRequest,
    ProcessScoreBatchRequest,
    ReplaceModelConfigEntry,
    ReplaceModelConfigsRequest,
    ScoreBatchItem,
    UnitCounts,
)
from sie_server.queue_executor import QueueExecutor
from sie_server.types.inputs import AudioInput, InvalidInputError, Item

CANARY = "sk-canary-0e6b94d27a1f3c58"
UPSTREAM_MODEL = "acme/fake-everything"
LOCAL_MODEL = "acme/remote-everything"
_TASKS = """\
inputs:
  text: true
  image: true
tasks:
  encode:
    dense:
      dim: 32
    sparse:
      dim: 1000
    multivector:
      dim: 16
  score: {}
  extract: {}
"""
UPSTREAM_YAML = f"""\
sie_id: {UPSTREAM_MODEL}
package_backed: true
{_TASKS}profiles:
  default:
    adapter_path: sie_server.adapters.fake.adapter:FakeAdapter
    max_batch_tokens: 8192
    adapter_options:
      loadtime:
        synthetic_usage: true
"""
LOCAL_YAML = f"""\
sie_id: {LOCAL_MODEL}
remote_backed: true
{_TASKS}profiles:
  default:
    adapter_path: sie_server.adapters.remote.sie:SieUpstreamAdapter
    max_batch_tokens: 8192
    adapter_options:
      loadtime:
        upstream: fake-sie
        upstream_model: {UPSTREAM_MODEL}
"""


@pytest.fixture(autouse=True)
def _credential(upstream_credential: str) -> str:
    return upstream_credential


@pytest.fixture
def upstream_models(tmp_path: Path) -> Path:
    models = tmp_path / "upstream-models"
    models.mkdir()
    (models / "everything.yaml").write_text(UPSTREAM_YAML, encoding="utf-8")
    return models


def png() -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (2, 2), color=(200, 30, 30)).save(buffer, format="PNG")
    return buffer.getvalue()


def post_until_served(client: TestClient, path: str, body: dict[str, Any]) -> httpx.Response:
    """POST until the answer is not this server's own ``MODEL_LOADING`` while the profile loads."""
    deadline = time.monotonic() + 30
    while True:
        response = client.post(path, json=body, headers={"Accept": "application/json"})
        loading_here = response.status_code == 503 and "x-sie-served-by" not in response.headers
        if not loading_here or time.monotonic() > deadline:
            return response
        time.sleep(0.1)


def assert_served_remotely(response: httpx.Response) -> None:
    assert response.headers["x-sie-served-by"] == "remote"
    assert response.headers["x-sie-upstream"] == "fake-sie"


def test_every_encode_output_comes_back_from_the_upstream_through_the_sdk(
    sie_upstream: Callable[..., Any],
    remote_app: Callable[..., Any],
    serve_on_loopback: Callable[..., Any],
    upstream_models: Path,
) -> None:
    items: list[Any] = [{"text": "remote backends serve every output"}, {"images": [{"data": png(), "format": "png"}]}]
    outputs = ["dense", "sparse", "multivector"]
    with sie_upstream(models_dir=upstream_models) as upstream:
        with SIEClient(upstream.url) as direct:
            expected = direct.encode(UPSTREAM_MODEL, items, output_types=outputs)
        app = remote_app(upstream.url, extra_models={"everything.yaml": LOCAL_YAML})
        with serve_on_loopback(app) as local_url, SIEClient(local_url) as client:
            served = client.encode(LOCAL_MODEL, items, output_types=outputs)

    assert len(served) == len(expected) == 2
    for got, want in zip(served, expected, strict=True):
        np.testing.assert_allclose(got["dense"], want["dense"], rtol=1e-6)
        np.testing.assert_array_equal(got["sparse"]["indices"], want["sparse"]["indices"])
        np.testing.assert_allclose(got["sparse"]["values"], want["sparse"]["values"], rtol=1e-6)
        np.testing.assert_allclose(got["multivector"], want["multivector"], rtol=1e-6)
        assert got["multivector"].shape[1] == 16


def test_encode_usage_is_the_upstreams_count_from_one_upstream_request_per_item(
    sie_upstream: Callable[..., Any], remote_app: Callable[..., Any], upstream_models: Path
) -> None:
    texts = ["one", "two words", "three more words"]
    body = {"items": [{"text": text} for text in texts], "params": {"output_types": ["dense", "sparse"]}}
    with sie_upstream(models_dir=upstream_models) as upstream:
        with SIEClient(upstream.url) as direct:
            direct.encode(UPSTREAM_MODEL, {"text": "warm"})
        upstream.posted_paths.clear()
        with TestClient(remote_app(upstream.url, extra_models={"everything.yaml": LOCAL_YAML})) as client:
            response = post_until_served(client, f"/v1/encode/{LOCAL_MODEL}", body)

    assert response.status_code == 200, response.text
    assert response.json()["usage"] == {"input_tokens": sum(len(text) for text in texts), "images": 0}
    assert upstream.posted_paths == [f"/v1/encode/{UPSTREAM_MODEL}"] * len(texts)
    assert_served_remotely(response)


def test_scores_and_their_usage_come_back_from_the_upstream_on_score_and_rerank(
    sie_upstream: Callable[..., Any], remote_app: Callable[..., Any], upstream_models: Path
) -> None:
    query = {"text": "which document answers the question"}
    items = [
        {"id": "alpha", "text": "a relevant document"},
        {"id": "beta", "text": "an unrelated one"},
        {"id": "gamma", "text": "a third candidate"},
    ]
    with sie_upstream(models_dir=upstream_models) as upstream:
        with SIEClient(upstream.url) as direct:
            expected = direct.score(UPSTREAM_MODEL, query, items)
        upstream.posted_paths.clear()
        with TestClient(remote_app(upstream.url, extra_models={"everything.yaml": LOCAL_YAML})) as client:
            scored = post_until_served(client, f"/v1/score/{LOCAL_MODEL}", {"query": query, "items": items})
            reranked = client.post(
                "/v1/rerank",
                json={"model": LOCAL_MODEL, "query": query["text"], "documents": [item["text"] for item in items]},
            )

    assert scored.status_code == 200, scored.text
    body = scored.json()
    assert [entry["item_id"] for entry in body["scores"]] == [entry["item_id"] for entry in expected["scores"]]
    assert [entry["score"] for entry in body["scores"]] == pytest.approx(
        [entry["score"] for entry in expected["scores"]], rel=1e-6
    )
    assert body["usage"] == expected["usage"]
    assert reranked.status_code == 200, reranked.text
    positions = {item["id"]: index for index, item in enumerate(items)}
    assert [result["index"] for result in reranked.json()["results"]] == [
        positions[entry["item_id"]] for entry in expected["scores"]
    ]
    assert reranked.json()["usage"] == expected["usage"]
    assert upstream.posted_paths == [f"/v1/score/{UPSTREAM_MODEL}"] * 2
    assert_served_remotely(scored)
    assert_served_remotely(reranked)


def test_extractions_come_back_from_the_upstream_and_an_item_error_is_fixed_text(
    sie_upstream: Callable[..., Any],
    remote_app: Callable[..., Any],
    serve_on_loopback: Callable[..., Any],
    upstream_models: Path,
) -> None:
    image = png()
    items: list[Any] = [
        {"text": "Ada Lovelace wrote programs in London"},
        {"images": [{"data": image, "format": "png"}]},
        {"text": ""},
    ]
    labels = ["person", "place"]
    with sie_upstream(models_dir=upstream_models) as upstream:
        with SIEClient(upstream.url) as direct:
            expected = direct.extract(UPSTREAM_MODEL, items, labels=labels)
        app = remote_app(upstream.url, extra_models={"everything.yaml": LOCAL_YAML})
        with serve_on_loopback(app) as local_url, SIEClient(local_url) as client:
            served = client.extract(LOCAL_MODEL, items, labels=labels)
        with TestClient(remote_app(upstream.url, extra_models={"everything.yaml": LOCAL_YAML})) as client:
            response = post_until_served(
                client,
                f"/v1/extract/{LOCAL_MODEL}",
                {
                    "items": [items[0], {"images": [{"data": base64.b64encode(image).decode(), "format": "png"}]}],
                    "params": {"labels": labels},
                },
            )

    for key in ("entities", "classifications", "relations", "objects", "data"):
        assert [result.get(key) for result in served[:2]] == [result.get(key) for result in expected[:2]], key
    assert served[0]["entities"]
    assert served[0]["relations"]
    assert served[1]["objects"]
    assert expected[2]["error"]["code"] == "INVALID_INPUT"
    assert served[2]["error"] == {"code": "INVALID_INPUT", "message": "the upstream refused this item"}
    assert response.status_code == 200, response.text
    assert response.json()["usage"] == {"input_tokens": len(items[0]["text"])}
    assert_served_remotely(response)


def mock_upstream_app(
    monkeypatch: pytest.MonkeyPatch, remote_app: Callable[..., Any], answer: Callable[[httpx.Request], httpx.Response]
) -> Any:
    monkeypatch.setattr(
        remote_sie,
        "upstream_sync_client",
        lambda upstream: upstream_sync_client(upstream, transport=httpx.MockTransport(answer)),
    )
    return remote_app("https://sie.example.internal", extra_models={"everything.yaml": LOCAL_YAML})


class _Body(httpx.SyncByteStream):
    """A body that streams like a network response, rather than one read eagerly from bytes."""

    def __init__(self, content: bytes) -> None:
        self._content = content

    def __iter__(self) -> Iterator[bytes]:
        yield self._content


def streamed(status: int, content: bytes, content_type: str = "application/msgpack", **headers: str) -> httpx.Response:
    return httpx.Response(status, stream=_Body(content), headers={"Content-Type": content_type, **headers})


def error_answer(status: int, code: str, *, retry_after: str | None = None) -> httpx.Response:
    headers = {} if retry_after is None else {"Retry-After": retry_after}
    content = json.dumps({"detail": {"code": code, "message": f"upstream says {CANARY}"}}).encode()
    return streamed(status, content, "application/json", **headers)


def test_an_upstream_that_cannot_serve_yet_is_a_retryable_503_on_score_extract_and_rerank(
    monkeypatch: pytest.MonkeyPatch, remote_app: Callable[..., Any]
) -> None:
    app = mock_upstream_app(
        monkeypatch, remote_app, lambda _request: error_answer(503, "MODEL_LOADING", retry_after="4")
    )

    with TestClient(app) as client:
        scored = post_until_served(
            client, f"/v1/score/{LOCAL_MODEL}", {"query": {"text": "q"}, "items": [{"text": "a"}]}
        )
        extracted = post_until_served(client, f"/v1/extract/{LOCAL_MODEL}", {"items": [{"text": "a"}]})
        reranked = client.post("/v1/rerank", json={"model": LOCAL_MODEL, "query": "q", "documents": ["a"]})

    for response in (scored, extracted):
        assert response.status_code == 503, response.text
        assert response.json()["detail"]["code"] == "MODEL_LOADING"
    assert reranked.status_code == 503, reranked.text
    assert reranked.json()["error"]["code"] == "MODEL_LOADING"
    for response in (scored, extracted, reranked):
        assert response.headers["retry-after"] == "4"
        assert_served_remotely(response)
        assert CANARY not in response.text


def test_an_input_the_upstream_refuses_is_the_callers_error_on_rerank(
    monkeypatch: pytest.MonkeyPatch, remote_app: Callable[..., Any]
) -> None:
    app = mock_upstream_app(monkeypatch, remote_app, lambda _request: error_answer(400, "INPUT_TOO_LONG"))

    with TestClient(app) as client:
        post_until_served(client, f"/v1/score/{LOCAL_MODEL}", {"query": {"text": "q"}, "items": [{"text": "a"}]})
        reranked = client.post("/v1/rerank", json={"model": LOCAL_MODEL, "query": "q", "documents": ["a"]})

    assert reranked.status_code == 400, reranked.text
    assert reranked.json()["error"] == {
        "code": "INPUT_TOO_LONG",
        "message": "the upstream refused the input as too long (400 INPUT_TOO_LONG)",
        "type": "invalid_request_error",
        "param": None,
    }


@pytest.fixture
def adapter_over(monkeypatch: pytest.MonkeyPatch) -> Iterator[Callable[..., remote_sie.SieUpstreamAdapter]]:
    """Build a loaded adapter whose upstream answers through ``handler`` on a mock transport."""
    loaded: list[remote_sie.SieUpstreamAdapter] = []

    def build(
        handler: Callable[[httpx.Request], httpx.Response],
        *,
        max_concurrency: int = 4,
        upstream_model: str = UPSTREAM_MODEL,
    ) -> remote_sie.SieUpstreamAdapter:
        upstream = Upstream.model_validate(
            {
                "kind": "sie",
                "base_url": "https://sie.example.internal",
                "rate_cap": {"requests_per_minute": 600, "max_concurrency": max_concurrency},
            }
        )
        install_upstreams({"team-sie": upstream})
        monkeypatch.setattr(
            remote_sie,
            "upstream_sync_client",
            lambda upstream: upstream_sync_client(upstream, transport=httpx.MockTransport(handler)),
        )
        adapter = remote_sie.SieUpstreamAdapter(
            upstream="team-sie", upstream_model=upstream_model, dense_dim=4, sparse_dim=10, multivector_dim=3
        )
        adapter.load("cpu")
        loaded.append(adapter)
        return adapter

    yield build
    for adapter in loaded:
        adapter.unload()


class Recorder:
    """An upstream that records each request body and answers with ``answer(body)``."""

    def __init__(self, answer: Callable[[dict[str, Any]], Any]) -> None:
        self.bodies: list[dict[str, Any]] = []
        self.paths: list[str] = []
        self._answer = answer
        self._lock = threading.Lock()

    def __call__(self, request: httpx.Request) -> httpx.Response:
        body = unpackb(request.content, numeric_arrays=False)
        with self._lock:
            self.bodies.append(body)
            self.paths.append(request.url.path)
        answer = self._answer(body)
        if isinstance(answer, httpx.Response):
            return answer
        return streamed(200, packb(answer))


def dense_answer(values: list[float], usage: Any = None) -> dict[str, Any]:
    vector = np.asarray(values, dtype=np.float32)
    body: dict[str, Any] = {"model": UPSTREAM_MODEL, "items": [{"dense": {"dims": 4, "values": vector}}]}
    if usage is not None:
        body["usage"] = usage
    return body


def test_each_item_is_one_upstream_request_with_its_own_count(
    adapter_over: Callable[..., remote_sie.SieUpstreamAdapter],
) -> None:
    recorder = Recorder(lambda body: dense_answer([1, 0, 0, 0], {"input_tokens": len(body["items"][0]["text"])}))
    adapter = adapter_over(recorder)

    output = adapter.encode([Item(text="a"), Item(text="bbb"), Item(text="cc")], ["dense"])

    assert sorted(body["items"][0]["text"] for body in recorder.bodies) == ["a", "bbb", "cc"]
    assert all(len(body["items"]) == 1 for body in recorder.bodies)
    assert output.extra["input_token_counts"] == [1, 3, 2]
    assert "input_image_counts" not in output.extra
    assert output.dense is not None
    assert output.dense.shape == (3, 4)


def test_an_answer_without_usage_reports_no_counts(adapter_over: Callable[..., remote_sie.SieUpstreamAdapter]) -> None:
    adapter = adapter_over(Recorder(lambda body: dense_answer([1, 0, 0, 0])))

    output = adapter.encode([Item(text="a")], ["dense"])

    assert "input_token_counts" not in output.extra


@pytest.mark.parametrize("max_concurrency", [3, 32])
def test_items_are_sent_concurrently_up_to_the_bound(
    adapter_over: Callable[..., remote_sie.SieUpstreamAdapter], max_concurrency: int
) -> None:
    lock = threading.Lock()
    current = peak = 0

    def answer(_body: dict[str, Any]) -> dict[str, Any]:
        nonlocal current, peak
        with lock:
            current += 1
            peak = max(peak, current)
        time.sleep(0.1)
        with lock:
            current -= 1
        return dense_answer([1, 0, 0, 0])

    adapter = adapter_over(Recorder(answer), max_concurrency=max_concurrency)

    adapter.encode([Item(text=str(index)) for index in range(16)], ["dense"])

    assert peak == min(max_concurrency, MAX_REQUESTS_IN_FLIGHT)


def test_after_a_failure_no_further_request_is_started(
    adapter_over: Callable[..., remote_sie.SieUpstreamAdapter],
) -> None:
    recorder = Recorder(lambda _body: error_answer(422, "INVALID_INPUT"))
    adapter = adapter_over(recorder, max_concurrency=1)

    with pytest.raises(InvalidInputError, match="the upstream refused the input"):
        adapter.encode([Item(text="a"), Item(text="b"), Item(text="c")], ["dense"])

    assert len(recorder.bodies) == 1


def test_options_this_server_consumes_stay_here(adapter_over: Callable[..., remote_sie.SieUpstreamAdapter]) -> None:
    recorder = Recorder(lambda _body: dense_answer([1, 0, 0, 0]))
    adapter = adapter_over(recorder)

    adapter.encode(
        [Item(text="a")],
        ["dense"],
        instruction="find the passage",
        is_query=True,
        options={
            "output_dtype": "binary",
            "muvera": {"k_sim": 3},
            "profile": "other",
            "lora": "adapter",
            "lora_id": "adapter",
            "output_types": ["sparse"],
            "instruction": "ignored",
            "is_query": False,
            "query_template": "query: {text}",
        },
    )

    assert recorder.bodies[0]["params"] == {
        "output_types": ["dense"],
        "output_dtype": "float32",
        "options": {"query_template": "query: {text}", "is_query": True},
        "instruction": "find the passage",
    }


def test_text_and_images_are_forwarded_and_other_inputs_refused_before_sending(
    adapter_over: Callable[..., remote_sie.SieUpstreamAdapter],
) -> None:
    recorder = Recorder(lambda _body: dense_answer([1, 0, 0, 0]))
    adapter = adapter_over(recorder)

    not_bytes: Any = {"data": "not bytes", "format": "png"}
    adapter.encode([Item(id="caller-id", text="caption", images=[{"data": b"\x89PNG", "format": "png"}])], ["dense"])
    for refused in (
        Item(text="a", audio=AudioInput(data=b"RIFF")),
        Item(video={"data": b"\0", "format": "mp4"}),
        Item(document={"data": b"%PDF", "format": "pdf"}),
        Item(images=[not_bytes]),
    ):
        with pytest.raises(InvalidInputError):
            adapter.encode([Item(text="sent first?"), refused], ["dense"])

    assert recorder.bodies == [
        {
            "items": [{"text": "caption", "images": [{"data": b"\x89PNG", "format": "png"}]}],
            "params": {"output_types": ["dense"], "output_dtype": "float32", "options": {"is_query": False}},
        }
    ]


def sparse_and_multivector_answer(item: dict[str, Any], usage: Any = None) -> dict[str, Any]:
    body: dict[str, Any] = {"model": UPSTREAM_MODEL, "items": [item]}
    if usage is not None:
        body["usage"] = usage
    return body


def good_item() -> dict[str, Any]:
    return {
        "sparse": {"indices": np.array([1, 7], dtype=np.int32), "values": np.array([0.5, 0.25], dtype=np.float32)},
        "multivector": {"values": np.ones((2, 3), dtype=np.float32)},
    }


def test_sparse_and_multivector_outputs_and_image_counts_come_back(
    adapter_over: Callable[..., remote_sie.SieUpstreamAdapter],
) -> None:
    usage = {"input_tokens": 5, "images": 1}
    adapter = adapter_over(Recorder(lambda _body: sparse_and_multivector_answer(good_item(), usage)))

    output = adapter.encode([Item(images=[{"data": b"\x89PNG", "format": "png"}])], ["sparse", "multivector"])

    assert output.dense is None
    assert output.sparse is not None
    assert output.sparse[0].indices.tolist() == [1, 7]
    assert output.sparse[0].indices.dtype == np.int32
    assert output.sparse[0].values.tolist() == [0.5, 0.25]
    assert output.multivector is not None
    assert output.multivector[0].shape == (2, 3)
    assert output.extra == {"input_token_counts": [5], "input_image_counts": [1]}


def with_field(field: str, value: Any) -> dict[str, Any]:
    item = good_item()
    item[field] = value
    return item


@pytest.mark.parametrize(
    ("item", "reason"),
    [
        (with_field("sparse", None), "without a sparse vector"),
        (
            with_field("sparse", {"indices": np.array([1.0]), "values": np.array([0.5], dtype=np.float32)}),
            "without a sparse vector",
        ),
        (
            with_field(
                "sparse", {"indices": np.array([1, 2], dtype=np.int32), "values": np.array([0.5], dtype=np.float32)}
            ),
            "without a sparse vector",
        ),
        (
            with_field(
                "sparse", {"indices": np.array([10], dtype=np.int32), "values": np.array([1], dtype=np.float32)}
            ),
            "outside the model's dimensions",
        ),
        (
            with_field(
                "sparse", {"indices": np.array([-1], dtype=np.int32), "values": np.array([1], dtype=np.float32)}
            ),
            "outside the model's dimensions",
        ),
        (
            with_field(
                "sparse", {"indices": np.array([1], dtype=np.int32), "values": np.array([np.inf], dtype=np.float32)}
            ),
            "non-finite",
        ),
        (with_field("multivector", {"values": np.ones(3, dtype=np.float32)}), "without a multivector"),
        (with_field("multivector", {"values": np.ones((2, 3), dtype=np.int32)}), "without a multivector"),
        (with_field("multivector", {"values": np.ones((2, 5), dtype=np.float32)}), "5-dimensional token vectors"),
        (with_field("multivector", {"values": np.full((1, 3), np.nan, dtype=np.float32)}), "non-finite"),
    ],
    ids=[
        "no-sparse",
        "float-indices",
        "lengths-differ",
        "index-past-the-vocabulary",
        "negative-index",
        "infinite-weight",
        "flat-multivector",
        "integer-multivector",
        "wrong-token-dimension",
        "nan-multivector",
    ],
)
def test_a_malformed_vector_is_an_error(
    adapter_over: Callable[..., remote_sie.SieUpstreamAdapter], item: dict[str, Any], reason: str
) -> None:
    adapter = adapter_over(Recorder(lambda _body: sparse_and_multivector_answer(item)))

    with pytest.raises(remote_http.RemoteUpstreamError, match=reason):
        adapter.encode([Item(text="a")], ["sparse", "multivector"])


@pytest.mark.parametrize(
    "usage",
    [
        {"input_tokens": -1},
        {"input_tokens": True},
        {"input_tokens": 1.5},
        {"input_tokens": 1 << 40},
        {"images": 2},
        {"input_tokens": 3, "images": "two"},
        {"input_tokens": 3, "input_tokens_details": {"content_tokens": 4}},
        {"input_tokens": 3, "input_tokens_details": 3},
        "lots",
    ],
    ids=[
        "negative",
        "bool",
        "float",
        "huge",
        "no-input-tokens",
        "bad-images",
        "content-over-total",
        "bad-details",
        "str",
    ],
)
def test_malformed_usage_is_an_error(adapter_over: Callable[..., remote_sie.SieUpstreamAdapter], usage: Any) -> None:
    adapter = adapter_over(Recorder(lambda _body: dense_answer([1, 0, 0, 0], usage)))

    with pytest.raises(remote_http.RemoteUpstreamError, match="malformed usage"):
        adapter.encode([Item(text="a")], ["dense"])


def score_answer(body: dict[str, Any]) -> dict[str, Any]:
    """Score each document by its position, sorted best first as SIE answers, with per-request usage."""
    ids = [item["id"] for item in body["items"]]
    ranked = sorted(ids, key=int, reverse=True)
    return {
        "model": UPSTREAM_MODEL,
        "scores": [{"item_id": item_id, "score": float(item_id), "rank": rank} for rank, item_id in enumerate(ranked)],
        "usage": {"input_tokens": 10 * len(ids), "input_tokens_details": {"content_tokens": len(ids)}, "images": 0},
    }


def test_each_api_request_is_scored_by_one_upstream_request(
    adapter_over: Callable[..., remote_sie.SieUpstreamAdapter],
) -> None:
    recorder = Recorder(score_answer)
    adapter = adapter_over(recorder)
    first, second = Item(text="first query"), Item(text="second query")

    output = adapter.score_pairs(
        [first, first, second],
        [Item(id="caller-a", text="a"), Item(id="caller-b", text="b"), Item(id="caller-c", text="c")],
        instruction="rank by relevance",
        options={"profile": "other", "overflow_policy": "truncate"},
    )

    sent = sorted(recorder.bodies, key=lambda body: body["query"]["text"])
    assert sent == [
        {
            "query": {"text": "first query"},
            "items": [{"text": "a", "id": "0"}, {"text": "b", "id": "1"}],
            "instruction": "rank by relevance",
            "options": {"overflow_policy": "truncate"},
        },
        {
            "query": {"text": "second query"},
            "items": [{"text": "c", "id": "0"}],
            "instruction": "rank by relevance",
            "options": {"overflow_policy": "truncate"},
        },
    ]
    assert recorder.paths == [f"/v1/score/{UPSTREAM_MODEL}"] * 2
    assert output.scores.tolist() == [0.0, 1.0, 0.0]
    assert output.input_token_counts == [20, 0, 10]
    assert output.content_token_counts == [2, 0, 1]
    assert output.input_image_counts == [0, 0, 0]


@pytest.mark.parametrize(
    ("answer", "reason"),
    [
        ({"scores": [{"item_id": "0", "score": 1.0}]}, "different number of scores"),
        ({"usage": {"input_tokens": 1}}, "different number of scores"),
        ({"scores": [{"item_id": "0", "score": 1.0}, {"item_id": "7", "score": 1.0}]}, "do not match the documents"),
        ({"scores": [{"item_id": "0", "score": 1.0}, {"item_id": "0", "score": 1.0}]}, "do not match the documents"),
        ({"scores": [{"item_id": 0, "score": 1.0}, {"item_id": 1, "score": 1.0}]}, "do not match the documents"),
        ({"scores": [{"item_id": "0", "score": float("nan")}, {"item_id": "1", "score": 1.0}]}, "not a finite number"),
        ({"scores": [{"item_id": "0", "score": True}, {"item_id": "1", "score": 1.0}]}, "not a finite number"),
        ({"scores": [{"item_id": "0"}, {"item_id": "1", "score": 1.0}]}, "not a finite number"),
    ],
    ids=["short", "no-scores", "unknown-id", "repeated-id", "integer-id", "nan", "bool", "missing-score"],
)
def test_scores_that_do_not_match_the_documents_are_an_error(
    adapter_over: Callable[..., remote_sie.SieUpstreamAdapter], answer: dict[str, Any], reason: str
) -> None:
    adapter = adapter_over(Recorder(lambda _body: answer))
    query = Item(text="q")

    with pytest.raises(remote_http.RemoteUpstreamError, match=reason):
        adapter.score_pairs([query, query], [Item(text="a"), Item(text="b")])


def extract_answer(item: dict[str, Any], usage: Any = None) -> dict[str, Any]:
    body: dict[str, Any] = {"model": UPSTREAM_MODEL, "items": [item]}
    if usage is not None:
        body["usage"] = usage
    return body


ENTITY = {"text": "Ada", "label": "person", "score": 0.9, "start": 0, "end": 3, "bbox": None}


def test_an_extraction_and_its_usage_come_back(adapter_over: Callable[..., remote_sie.SieUpstreamAdapter]) -> None:
    item = {
        "entities": [ENTITY, {"text": "box", "label": "region", "score": np.float32(0.5), "bbox": [1, 2, 3, 4]}],
        "relations": [{"head": "Ada", "tail": "London", "relation": "lived_in", "score": 0.7}],
        "classifications": [{"label": "biography", "score": 0.8}],
        "objects": [{"label": "cat", "score": 0.6, "bbox": [0, 0, 10, 12]}],
        "data": {"pages": [{"number": 1, "words": ["Ada"]}], "ratio": np.float64(0.5), "ok": True, "none": None},
    }
    recorder = Recorder(lambda body: extract_answer(item, {"input_tokens": len(body["items"][0]["text"])}))
    adapter = adapter_over(recorder)

    output = adapter.extract(
        [Item(text="Ada"), Item(text="Ada Lovelace")],
        labels=["person"],
        output_schema={"type": "object"},
        instruction="find people",
        options={"threshold": 0.4, "profile": "other"},
    )

    assert all(
        body["params"]
        == {
            "labels": ["person"],
            "output_schema": {"type": "object"},
            "instruction": "find people",
            "options": {"threshold": 0.4},
        }
        for body in recorder.bodies
    )
    assert output.entities[0] == [
        ENTITY,
        {"text": "box", "label": "region", "score": 0.5, "start": None, "end": None, "bbox": [1, 2, 3, 4]},
    ]
    assert output.relations == [item["relations"]] * 2
    assert output.classifications == [item["classifications"]] * 2
    assert output.objects == [item["objects"]] * 2
    assert output.data == [{"pages": [{"number": 1, "words": ["Ada"]}], "ratio": 0.5, "ok": True, "none": None}] * 2
    assert output.errors is None
    assert output.input_token_counts == [3, 12]


def nested(depth: int) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for _ in range(depth):
        value = {"next": value}
    return value


@pytest.mark.parametrize(
    "item",
    [
        {"entities": {"text": "Ada"}},
        {"entities": [{"label": "person", "score": 0.9}]},
        {"entities": [{**ENTITY, "score": float("nan")}]},
        {"entities": [{**ENTITY, "start": -1}]},
        {"entities": [{**ENTITY, "end": 2.5}]},
        {"entities": [{**ENTITY, "bbox": [1, 2, 3]}]},
        {"entities": [{**ENTITY, "bbox": [1.5, 2, 3, 4]}]},
        {"relations": [{"tail": "London", "relation": "lived_in", "score": 0.7}]},
        {"classifications": [{"label": "biography", "score": "high"}]},
        {"objects": [{"label": "cat", "score": 0.6}]},
        {"data": [1, 2]},
        {"data": {"blob": b"\x00\x01"}},
        {"data": {"ratio": float("inf")}},
        {"data": nested(80)},
    ],
    ids=[
        "entities-not-a-list",
        "entity-without-text",
        "nan-score",
        "negative-offset",
        "fractional-offset",
        "short-bbox",
        "fractional-bbox",
        "relation-without-head",
        "text-score",
        "object-without-bbox",
        "data-not-an-object",
        "bytes-in-data",
        "infinite-in-data",
        "data-too-deep",
    ],
)
def test_a_malformed_extraction_is_an_error(
    adapter_over: Callable[..., remote_sie.SieUpstreamAdapter], item: dict[str, Any]
) -> None:
    adapter = adapter_over(Recorder(lambda _body: extract_answer(item)))

    with pytest.raises(remote_http.RemoteUpstreamError, match="malformed extraction"):
        adapter.extract([Item(text="Ada")])


@pytest.mark.parametrize(
    ("error", "code", "message"),
    [
        ({"code": "INVALID_INPUT", "message": CANARY}, "INVALID_INPUT", "the upstream refused this item"),
        ({"code": "INPUT_TOO_LONG", "message": CANARY}, "INPUT_TOO_LONG", "the upstream refused this item as too long"),
        (
            {"code": f"LEAK {CANARY}", "message": CANARY},
            "INFERENCE_ERROR",
            "the upstream could not extract from this item",
        ),
        (CANARY, "INFERENCE_ERROR", "the upstream could not extract from this item"),
    ],
    ids=["invalid-input", "too-long", "unknown-code", "not-an-object"],
)
def test_an_item_the_upstream_could_not_extract_from_is_a_known_code_and_fixed_text(
    adapter_over: Callable[..., remote_sie.SieUpstreamAdapter], error: Any, code: str, message: str
) -> None:
    adapter = adapter_over(Recorder(lambda _body: extract_answer({"entities": [], "error": error})))

    output = adapter.extract([Item(text="Ada")])

    assert output.errors is not None
    assert (output.errors[0].code, output.errors[0].message) == (code, message)


@pytest.mark.parametrize(
    ("primitive", "call"),
    [
        ("an encode", lambda adapter: adapter.encode([Item(text="a")], ["dense"])),
        ("a score", lambda adapter: adapter.score_pairs([Item(text="q")], [Item(text="a")])),
        ("an extract", lambda adapter: adapter.extract([Item(text="a")])),
    ],
    ids=["encode", "score", "extract"],
)
def test_a_model_id_that_escapes_the_route_is_never_sent(
    adapter_over: Callable[..., remote_sie.SieUpstreamAdapter],
    primitive: str,
    call: Callable[[remote_sie.SieUpstreamAdapter], Any],
) -> None:
    recorder = Recorder(lambda _body: {})
    adapter = adapter_over(recorder, upstream_model="../../admin")

    with pytest.raises(remote_http.RemoteUpstreamError, match=f"does not form {primitive} path"):
        call(adapter)

    assert recorder.bodies == []


@pytest.mark.parametrize(
    ("call", "response"),
    [
        (lambda adapter: adapter.score_pairs([Item(text="q")], [Item(text="a")]), "a score"),
        (lambda adapter: adapter.extract([Item(text="a")]), "an extract"),
    ],
    ids=["score", "extract"],
)
def test_an_undecodable_answer_is_an_error(
    adapter_over: Callable[..., remote_sie.SieUpstreamAdapter],
    call: Callable[[remote_sie.SieUpstreamAdapter], Any],
    response: str,
) -> None:
    adapter = adapter_over(lambda _request: streamed(200, b"\xc1"))

    with pytest.raises(remote_http.RemoteUpstreamError, match=f"not {response} response"):
        call(adapter)


def test_the_loader_gives_the_declared_sparse_and_multivector_dims_to_constructors_that_take_them(
    tmp_path: Path,
) -> None:
    remote = load_adapter(ModelConfig.model_validate(_yaml(LOCAL_YAML)), tmp_path, device="cpu")
    fake = load_adapter(ModelConfig.model_validate(_yaml(UPSTREAM_YAML)), tmp_path, device="cpu")

    for adapter in (remote, fake):
        assert (adapter.dims.dense, adapter.dims.sparse, adapter.dims.multivector) == (32, 1000, 16)


def _yaml(text: str) -> dict[str, Any]:
    return yaml.safe_load(text)


async def wait_until_ready(executor: QueueExecutor, model_id: str) -> None:
    deadline = time.monotonic() + 30
    while (state := await executor.ensure_model_ready(model_id)) != "ready":
        assert state in {"loading_started", "loading_in_progress"}, state
        assert time.monotonic() < deadline, "the remote profile did not load"
        await asyncio.sleep(0.05)


def batch_item_fields(request_id: str) -> dict[str, Any]:
    return {
        "work_item_id": f"{request_id}.0",
        "request_id": request_id,
        "item_index": 0,
        "total_items": 1,
        "timestamp": time.time(),
    }


async def test_every_queued_work_item_is_metered_by_its_own_upstream_request(
    sie_upstream: Callable[..., Any], upstream_models: Path, upstream_credential_env: str
) -> None:
    with sie_upstream(models_dir=upstream_models) as upstream:
        with SIEClient(upstream.url) as direct:
            direct.encode(UPSTREAM_MODEL, {"text": "warm"})
        upstream.posted_paths.clear()
        install_upstreams(
            {
                "fake-sie": Upstream.model_validate(
                    {
                        "kind": "sie",
                        "base_url": upstream.url,
                        "api_key_secret": upstream_credential_env,
                        "rate_cap": {"requests_per_minute": 600, "max_concurrency": 8},
                    }
                )
            }
        )
        executor = QueueExecutor(ModelRegistry(models_dir=None))
        applied = await executor.replace_model_configs(
            ReplaceModelConfigsRequest(
                bundle_id="remote",
                epoch=1,
                bundle_config_hash="",
                models=[ReplaceModelConfigEntry(model_id=LOCAL_MODEL, model_config=LOCAL_YAML)],
            )
        )
        await wait_until_ready(executor, LOCAL_MODEL)
        encoded = await executor.process_encode_batch(
            ProcessEncodeBatchRequest(
                model_id=LOCAL_MODEL,
                items=[
                    EncodeBatchItem(
                        **batch_item_fields(request_id),
                        item={"text": text},
                        output_types=["dense"],
                        bundle_config_hash=applied.bundle_config_hash,
                    )
                    for request_id, text in (("e1", "short"), ("e2", "a longer text"))
                ],
            )
        )
        scored = await executor.process_score_batch(
            ProcessScoreBatchRequest(
                model_id=LOCAL_MODEL,
                items=[
                    ScoreBatchItem(
                        **batch_item_fields("s1"),
                        query_item={"text": "first query"},
                        score_items=[{"text": "x"}, {"text": "yy"}],
                    ),
                    ScoreBatchItem(
                        **batch_item_fields("s2"), query_item={"text": "second"}, score_items=[{"text": "zzz"}]
                    ),
                ],
            )
        )
        extracted = await executor.process_extract_batch(
            ProcessExtractBatchRequest(
                model_id=LOCAL_MODEL,
                items=[
                    ExtractBatchItem(
                        **batch_item_fields(request_id),
                        item={"text": text},
                        labels=["person"],
                        bundle_config_hash=applied.bundle_config_hash,
                    )
                    for request_id, text in (("x1", "Ada"), ("x2", "Ada Lovelace"))
                ],
            )
        )

    outcomes = [*encoded.outcomes, *scored.outcomes, *extracted.outcomes]
    assert all(outcome.disposition == "publish_and_ack" for outcome in outcomes), [o.error for o in outcomes]
    assert [outcome.units for outcome in encoded.outcomes] == [
        UnitCounts(input_tokens=len("short")),
        UnitCounts(input_tokens=len("a longer text")),
    ]
    assert [outcome.units for outcome in scored.outcomes] == [
        UnitCounts(input_tokens=2 * len("first query") + len("x") + len("yy"), pairs=2),
        UnitCounts(input_tokens=len("second") + len("zzz"), pairs=1),
    ]
    assert [outcome.units.input_tokens for outcome in extracted.outcomes if outcome.units is not None] == [3, 12]
    assert sorted(upstream.posted_paths) == sorted(
        [f"/v1/encode/{UPSTREAM_MODEL}"] * 2
        + [f"/v1/score/{UPSTREAM_MODEL}"] * 2
        + [f"/v1/extract/{UPSTREAM_MODEL}"] * 2
    )
