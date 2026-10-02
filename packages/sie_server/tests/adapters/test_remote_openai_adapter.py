"""A remote profile on an upstream of kind openai: dense encode and rerank.

The upstream is a small OpenAI-compatible app on loopback that records every
call it receives, so each request crosses the real wire format. A local app
serves the remote-backed model ``acme/openai-remote`` through it.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import time
from collections.abc import Callable, Iterator
from contextlib import AbstractContextManager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx
import numpy as np
import pytest
import yaml
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response
from fastapi.testclient import TestClient
from sie_server.adapters.remote._http import RemoteUpstreamError
from sie_server.adapters.remote.openai import OpenAIUpstreamAdapter
from sie_server.app.app_factory import AppFactory
from sie_server.app.app_state_config import AppStateConfig
from sie_server.config.upstreams import Upstream, UpstreamConfigError, install_upstreams
from sie_server.types.inputs import InvalidInputError, Item

DIM = 8
MODEL = "acme/openai-remote"
ADAPTER = "sie_server.adapters.remote.openai:OpenAIUpstreamAdapter"
UPSTREAM_MODEL = "org/embed-rerank-v1"
OPERATOR_FIELDS = {"provider": {"data_collection": "deny"}}
CANARY = "upstream-detail-5a1f9c"
PNG_B64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkYAAAAAYAAjCB0C8AAAAASUVORK5CYII="
RATE_CAP = {"requests_per_minute": 600, "max_concurrency": 8}


def vector(text: str) -> list[float]:
    return [byte / 255 for byte in hashlib.sha256(text.encode()).digest()[:DIM]]


def relevance(query: str, document: str) -> float:
    words = set(query.split())
    return len(words & set(document.split())) / len(words) + len(document) / 1000


def word_count(*texts: str) -> int:
    return sum(len(text.split()) for text in texts)


def embedding_bodies(texts: list[str]) -> list[dict[str, Any]]:
    return [{"model": UPSTREAM_MODEL, "input": [text], "encoding_format": "float", **OPERATOR_FIELDS} for text in texts]


@dataclass
class FakeOpenAI:
    """An OpenAI-compatible upstream that records each call and answers from a queue first.

    ``delays`` holds seconds to wait before answering an embeddings input, so
    concurrent calls can finish out of order.
    """

    url: str = ""
    calls: list[dict[str, Any]] = field(default_factory=list)
    queued: list[Response] = field(default_factory=list)
    delays: dict[str, float] = field(default_factory=dict)

    def app(self) -> FastAPI:
        app = FastAPI()

        async def record(request: Request) -> dict[str, Any]:
            body = await request.json()
            self.calls.append(
                {"path": request.url.path, "authorization": request.headers.get("authorization"), "body": body}
            )
            return body

        @app.post("/v1/embeddings")
        async def embeddings(request: Request) -> Response:
            body = await record(request)
            if self.queued:
                return self.queued.pop(0)
            texts = body["input"]
            await asyncio.sleep(max((self.delays.get(text, 0.0) for text in texts), default=0.0))
            data = [{"object": "embedding", "index": i, "embedding": vector(text)} for i, text in enumerate(texts)]
            usage = {"prompt_tokens": word_count(*texts), "total_tokens": word_count(*texts)}
            return JSONResponse({"object": "list", "data": data, "model": body["model"], "usage": usage})

        @app.post("/v1/rerank")
        async def rerank(request: Request) -> Response:
            body = await record(request)
            if self.queued:
                return self.queued.pop(0)
            documents = body["documents"]
            results = sorted(
                ({"index": i, "relevance_score": relevance(body["query"], doc)} for i, doc in enumerate(documents)),
                key=lambda result: -result["relevance_score"],
            )
            usage = {"total_tokens": word_count(body["query"], *documents)}
            return JSONResponse({"results": results[: body["top_n"]], "usage": usage})

        return app


@pytest.fixture
def fake_openai(
    serve_on_loopback: Callable[[FastAPI], AbstractContextManager[str]], _offline_apps: None
) -> Iterator[FakeOpenAI]:
    fake = FakeOpenAI()
    with serve_on_loopback(fake.app()) as url:
        fake.url = url
        yield fake


def upstream_spec(url: str, *, endpoints: tuple[str, ...] = ("embeddings", "rerank"), **fields: Any) -> dict[str, Any]:
    return {
        "kind": "openai",
        "base_url": f"{url}/v1",
        "endpoints": list(endpoints),
        "set_params": OPERATOR_FIELDS,
        "strip_params": ["user"],
        "rate_cap": RATE_CAP,
        **fields,
    }


def remote_model(*, runtime: dict[str, Any] | None = None) -> dict[str, Any]:
    return {
        "sie_id": MODEL,
        "remote_backed": True,
        "inputs": {"text": True, "image": True},
        "tasks": {"encode": {"dense": {"dim": DIM}, "sparse": {"dim": 1000}}, "score": {}},
        "profiles": {
            "default": {
                "adapter_path": ADAPTER,
                "max_batch_tokens": 8192,
                "adapter_options": {
                    "loadtime": {"upstream": "open-host", "upstream_model": UPSTREAM_MODEL},
                    "runtime": runtime or {},
                },
            }
        },
    }


@pytest.fixture
def openai_app(
    tmp_path: Path, upstream_credential: str, upstream_credential_env: str, _offline_apps: None
) -> Callable[..., FastAPI]:
    """Build a local app that serves ``acme/openai-remote`` through the upstream at a URL."""
    _ = upstream_credential

    def build(
        upstream_url: str,
        *,
        endpoints: tuple[str, ...] = ("embeddings", "rerank"),
        runtime: dict[str, Any] | None = None,
    ) -> FastAPI:
        models = tmp_path / "models"
        models.mkdir(exist_ok=True)
        (models / "openai-remote.yaml").write_text(yaml.safe_dump(remote_model(runtime=runtime)), encoding="utf-8")
        upstreams = tmp_path / "upstreams.yaml"
        spec = upstream_spec(upstream_url, endpoints=endpoints, api_key_secret=upstream_credential_env)
        upstreams.write_text(yaml.safe_dump({"upstreams": {"open-host": spec}}), encoding="utf-8")
        return AppFactory.create_app(
            AppStateConfig(models_dir=str(models), device="cpu", upstreams_file=str(upstreams))
        )

    return build


def post(
    client: TestClient, fake: FakeOpenAI, path: str, body: dict[str, Any], headers: dict[str, str] | None = None
) -> httpx.Response:
    """POST until the local model has loaded: a 503 that sent nothing upstream is the local load, for at most 30 s."""
    deadline = time.monotonic() + 30
    while True:
        calls_before = len(fake.calls)
        response = client.post(path, json=body, headers={"Accept": "application/json", **(headers or {})})
        if response.status_code != 503 or len(fake.calls) > calls_before or time.monotonic() > deadline:
            return response
        time.sleep(0.1)


def error_code(response: httpx.Response) -> str:
    body = response.json()
    envelope = body.get("detail") or body.get("error") or {}
    return envelope.get("code", "")


@pytest.fixture(autouse=True)
def _credential(upstream_credential: str) -> str:
    return upstream_credential


def test_dense_encode_sends_one_embeddings_call_per_item(
    fake_openai: FakeOpenAI,
    openai_app: Callable[..., FastAPI],
    upstream_credential: str,
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.DEBUG)
    fake_openai.delays = {"red apple": 0.3}
    with TestClient(openai_app(fake_openai.url)) as client:
        response = post(
            client,
            fake_openai,
            f"/v1/encode/{MODEL}",
            {"items": [{"text": "red apple"}, {"text": "green pear"}, {"text": "ripe plum"}]},
            {"Authorization": "Bearer caller-token-1"},
        )
        listed = client.get(f"/v1/models/{MODEL}")

    assert response.status_code == 200, response.text
    served = [item["dense"]["values"] for item in response.json()["items"]]
    np.testing.assert_allclose(served, [vector("red apple"), vector("green pear"), vector("ripe plum")], rtol=1e-6)
    assert response.json()["usage"]["input_tokens"] == 6
    assert response.headers["x-sie-served-by"] == "remote"
    assert response.headers["x-sie-upstream"] == "open-host"
    assert listed.json()["routing"] == {"policy": "remote_only", "upstream_kind": "openai"}
    assert {call["path"] for call in fake_openai.calls} == {"/v1/embeddings"}
    assert {call["authorization"] for call in fake_openai.calls} == {f"Bearer {upstream_credential}"}
    assert sorted((call["body"] for call in fake_openai.calls), key=str) == sorted(
        embedding_bodies(["red apple", "green pear", "ripe plum"]), key=str
    )
    assert upstream_credential not in response.text
    assert upstream_credential not in caplog.text


def test_the_caller_cannot_add_fields_to_the_upstream_call(
    fake_openai: FakeOpenAI, openai_app: Callable[..., FastAPI]
) -> None:
    with TestClient(openai_app(fake_openai.url)) as client:
        response = post(
            client,
            fake_openai,
            "/v1/embeddings",
            {
                "model": MODEL,
                "input": ["red apple"],
                "user": "caller-7",
                "dimensions": DIM,
                "provider": {"data_collection": "allow"},
                "extra_body": {"top_k": 3},
            },
        )

    assert response.status_code == 200, response.text
    np.testing.assert_allclose(response.json()["data"][0]["embedding"], vector("red apple"), rtol=1e-6)
    assert response.json()["usage"]["prompt_tokens"] == 2
    assert [call["body"] for call in fake_openai.calls] == embedding_bodies(["red apple"])


def test_the_profile_templates_shape_the_text_the_upstream_embeds(
    fake_openai: FakeOpenAI, openai_app: Callable[..., FastAPI]
) -> None:
    app = openai_app(fake_openai.url, runtime={"query_template": "query: {text}", "doc_template": "passage: {text}"})
    with TestClient(app) as client:
        query = post(
            client,
            fake_openai,
            f"/v1/encode/{MODEL}",
            {"items": [{"text": "red apple"}], "params": {"options": {"is_query": True}}},
        )
        document = post(client, fake_openai, f"/v1/encode/{MODEL}", {"items": [{"text": "red apple"}]})

    assert query.status_code == 200, query.text
    assert document.status_code == 200, document.text
    assert [call["body"]["input"] for call in fake_openai.calls] == [["query: red apple"], ["passage: red apple"]]


@pytest.mark.parametrize(
    ("body", "reason"),
    [
        (
            {"items": [{"text": "red apple"}], "params": {"output_types": ["dense", "sparse"]}},
            "returns dense vectors only; this profile cannot produce sparse",
        ),
        ({"items": [{"images": [{"data": PNG_B64, "format": "png"}]}]}, "takes items of plain text only"),
        (
            {"items": [{"text": "red apple"}, {"text": "pear", "images": [{"data": PNG_B64, "format": "png"}]}]},
            "takes items of plain text only",
        ),
    ],
    ids=["sparse-output", "image-item", "one-item-with-an-image"],
)
def test_what_the_upstream_cannot_express_is_refused_before_anything_is_sent(
    fake_openai: FakeOpenAI, openai_app: Callable[..., FastAPI], body: dict[str, Any], reason: str
) -> None:
    with TestClient(openai_app(fake_openai.url)) as client:
        response = post(client, fake_openai, f"/v1/encode/{MODEL}", body)

    assert response.status_code == 400, response.text
    assert error_code(response) == "INVALID_INPUT"
    assert reason in response.text
    assert fake_openai.calls == []


def test_rerank_sends_one_call_per_request_and_keeps_each_documents_score(
    fake_openai: FakeOpenAI, openai_app: Callable[..., FastAPI]
) -> None:
    documents = {"pear": "a green pear", "pie": "a red apple pie", "wine": "red wine"}
    with TestClient(openai_app(fake_openai.url)) as client:
        response = post(
            client,
            fake_openai,
            f"/v1/score/{MODEL}",
            {"query": {"text": "red apple"}, "items": [{"id": key, "text": text} for key, text in documents.items()]},
        )

    assert response.status_code == 200, response.text
    scores = {entry["item_id"]: entry["score"] for entry in response.json()["scores"]}
    assert scores == pytest.approx({key: relevance("red apple", text) for key, text in documents.items()}, rel=1e-6)
    assert response.json()["usage"]["input_tokens"] == word_count("red apple", *documents.values())
    [call] = fake_openai.calls
    sent_documents = call["body"].pop("documents")
    assert sorted(sent_documents) == sorted(documents.values())
    assert call["body"] == {"model": UPSTREAM_MODEL, "query": "red apple", "top_n": 3, **OPERATOR_FIELDS}


def test_the_openai_rerank_route_asks_the_upstream_for_every_score(
    fake_openai: FakeOpenAI, openai_app: Callable[..., FastAPI]
) -> None:
    documents = ["a green pear", "a red apple pie", "red wine"]
    with TestClient(openai_app(fake_openai.url)) as client:
        response = post(
            client,
            fake_openai,
            "/v1/rerank",
            {"model": MODEL, "query": "red apple", "documents": documents, "top_n": 1},
        )

    assert response.status_code == 200, response.text
    [best] = response.json()["results"]
    assert best["index"] == 1
    assert best["relevance_score"] == pytest.approx(relevance("red apple", documents[1]), rel=1e-6)
    assert [call["body"]["top_n"] for call in fake_openai.calls] == [3]


def test_an_operation_the_upstream_does_not_declare_is_refused_before_sending(
    fake_openai: FakeOpenAI, openai_app: Callable[..., FastAPI]
) -> None:
    with TestClient(openai_app(fake_openai.url, endpoints=("embeddings",))) as client:
        response = post(
            client, fake_openai, f"/v1/score/{MODEL}", {"query": {"text": "red"}, "items": [{"text": "red wine"}]}
        )

    assert response.status_code == 400, response.text
    assert error_code(response) == "INVALID_INPUT"
    assert "declares no rerank endpoint" in response.text
    assert fake_openai.calls == []


@pytest.mark.parametrize(
    ("status", "headers", "retry_after"),
    [(503, {"Retry-After": "7"}, "7"), (429, {}, "5"), (502, {}, "5")],
    ids=["unavailable-with-retry-after", "rate-limited", "bad-gateway"],
)
def test_an_upstream_that_may_recover_is_a_retryable_503(
    fake_openai: FakeOpenAI,
    openai_app: Callable[..., FastAPI],
    status: int,
    headers: dict[str, str],
    retry_after: str,
) -> None:
    with TestClient(openai_app(fake_openai.url)) as client:
        warm = post(client, fake_openai, f"/v1/encode/{MODEL}", {"items": [{"text": "warm"}]})
        fake_openai.queued.append(
            JSONResponse({"error": {"message": CANARY, "type": "server_error"}}, status_code=status, headers=headers)
        )
        response = post(client, fake_openai, f"/v1/encode/{MODEL}", {"items": [{"text": "red apple"}]})

    assert warm.status_code == 200, warm.text
    assert response.status_code == 503, response.text
    assert error_code(response) == "QUEUE_FULL"
    assert response.headers["retry-after"] == retry_after
    assert response.headers["x-sie-upstream"] == "open-host"
    assert CANARY not in response.text


@pytest.mark.parametrize("status", [400, 401, 404, 422])
def test_an_upstream_refusal_is_a_server_error_that_repeats_nothing_it_sent(
    fake_openai: FakeOpenAI, openai_app: Callable[..., FastAPI], status: int, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.DEBUG)
    with TestClient(openai_app(fake_openai.url)) as client:
        warm = post(client, fake_openai, f"/v1/encode/{MODEL}", {"items": [{"text": "warm"}]})
        fake_openai.queued.append(
            JSONResponse(
                {"error": {"message": CANARY, "type": "invalid_request_error", "code": "context_length_exceeded"}},
                status_code=status,
            )
        )
        response = post(client, fake_openai, f"/v1/encode/{MODEL}", {"items": [{"text": "red apple"}]})

    assert warm.status_code == 200, warm.text
    assert response.status_code == 500, response.text
    assert error_code(response) == "INFERENCE_ERROR"
    assert CANARY not in response.text
    assert CANARY not in caplog.text


def load_adapter(url: str, *, dense_dim: int | None = DIM) -> OpenAIUpstreamAdapter:
    install_upstreams({"open-host": Upstream.model_validate(upstream_spec(url))})
    loaded = OpenAIUpstreamAdapter(upstream="open-host", upstream_model=UPSTREAM_MODEL, dense_dim=dense_dim)
    loaded.load("cpu")
    return loaded


@pytest.fixture
def adapter(fake_openai: FakeOpenAI) -> Iterator[OpenAIUpstreamAdapter]:
    loaded = load_adapter(fake_openai.url)
    yield loaded
    loaded.unload()
    install_upstreams({})


def test_the_pairs_of_each_request_go_in_one_rerank_call(
    adapter: OpenAIUpstreamAdapter, fake_openai: FakeOpenAI
) -> None:
    fruit, drink = Item(text="red apple"), Item(text="red wine")
    documents = ["a red apple pie", "a glass of red wine", "a green apple", "white wine"]

    output = adapter.score_pairs([fruit, drink, fruit, drink], [Item(text=doc) for doc in documents])

    sent = sorted((call["body"]["query"], call["body"]["documents"]) for call in fake_openai.calls)
    assert sent == [
        ("red apple", ["a red apple pie", "a green apple"]),
        ("red wine", ["a glass of red wine", "white wine"]),
    ]
    queries = ["red apple", "red wine", "red apple", "red wine"]
    np.testing.assert_allclose(output.scores, [relevance(q, d) for q, d in zip(queries, documents)], rtol=1e-6)
    assert output.input_token_counts == [
        word_count("red apple", "a red apple pie", "a green apple"),
        word_count("red wine", "a glass of red wine", "white wine"),
        0,
        0,
    ]


def test_a_rerank_instruction_leads_the_query(adapter: OpenAIUpstreamAdapter, fake_openai: FakeOpenAI) -> None:
    adapter.score(Item(text="red apple"), [Item(text="a red apple pie")], instruction="Find fruit:")

    assert fake_openai.calls[0]["body"]["query"] == "Find fruit: red apple"


def test_the_normalize_option_returns_unit_vectors(adapter: OpenAIUpstreamAdapter) -> None:
    output = adapter.encode([Item(text="red apple")], ["dense"], options={"normalize": True})

    assert output.dense is not None
    expected = np.asarray(vector("red apple"), dtype=np.float32)
    np.testing.assert_allclose(output.dense[0], expected / np.linalg.norm(expected), rtol=1e-6)


def test_a_count_is_recorded_only_when_every_call_reported_one(
    adapter: OpenAIUpstreamAdapter, fake_openai: FakeOpenAI
) -> None:
    fake_openai.queued.append(JSONResponse({"data": [{"index": 0, "embedding": vector("x")}]}))
    fake_openai.queued.append(JSONResponse({"results": [{"index": 0, "relevance_score": 0.5}]}))

    encoded = adapter.encode([Item(text="x")], ["dense"])
    scored = adapter.score_pairs([Item(text="q")], [Item(text="d")])
    reported = adapter.encode([Item(text="one two"), Item(text="three")], ["dense"])

    assert "input_token_counts" not in encoded.extra
    assert scored.input_token_counts is None
    assert reported.extra["input_token_counts"] == [2, 1]


def test_embeddings_of_different_widths_are_an_error(fake_openai: FakeOpenAI) -> None:
    unsized = load_adapter(fake_openai.url, dense_dim=None)
    fake_openai.queued.extend(
        [
            JSONResponse({"data": [{"index": 0, "embedding": [1.0, 2.0]}]}),
            JSONResponse({"data": [{"index": 0, "embedding": [1.0, 2.0, 3.0]}]}),
        ]
    )
    try:
        with pytest.raises(RemoteUpstreamError, match="embeddings of different widths"):
            unsized.encode([Item(text="a"), Item(text="b")], ["dense"])
    finally:
        unsized.unload()
        install_upstreams({})


def answer(content: str) -> Response:
    return Response(content=content, media_type="application/json")


ONE = ", ".join(["1"] * DIM)


@pytest.mark.parametrize(
    ("body", "reason"),
    [
        (answer('{"data": []}'), "different number of embeddings"),
        (
            answer(f'{{"data": [{{"index": 0, "embedding": [{ONE}]}}, {{"index": 1, "embedding": [{ONE}]}}]}}'),
            "different number",
        ),
        (answer("[]"), "not a JSON object"),
        (answer(f'{{"data": [{{"index": 1, "embedding": [{ONE}]}}]}}'), "an input it was not sent"),
        (answer(f'{{"data": [{{"index": "0", "embedding": [{ONE}]}}]}}'), "an input it was not sent"),
        (answer('{"data": [{"index": 0, "embedding": [[1, 2], [3]]}]}'), "not a list of numbers"),
        (answer('{"data": [{"index": 0, "embedding": "AAAAAAAAAAA="}]}'), "not a list of numbers"),
        (answer('{"data": [{"index": 0, "embedding": ["1", "2"]}]}'), "not a list of numbers"),
        (answer('{"data": [{"index": 0, "embedding": null}]}'), "not a list of numbers"),
        (answer('{"data": [{"index": 0, "embedding": [1, 2]}]}'), "2-dimensional"),
        (answer('{"data": [{"index": 0, "embedding": [NaN, 1, 1, 1, 1, 1, 1, 1]}]}'), "non-finite"),
        (
            answer(f'{{"data": [{{"index": 0, "embedding": [{ONE}]}}], "usage": {{"prompt_tokens": -1}}}}'),
            "malformed usage",
        ),
        (
            answer(f'{{"data": [{{"index": 0, "embedding": [{ONE}]}}], "usage": {{"prompt_tokens": "3"}}}}'),
            "malformed usage",
        ),
        (answer(f'{{"data": [{{"index": 0, "embedding": [{ONE}]}}], "usage": []}}'), "malformed usage"),
        (answer("<html>busy</html>"), "not JSON"),
    ],
    ids=[
        "no-data",
        "two-for-one",
        "not-an-object",
        "other-index",
        "string-index",
        "nested",
        "base64",
        "strings",
        "null",
        "wrong-width",
        "nan",
        "negative-usage",
        "string-usage",
        "usage-not-an-object",
        "not-json",
    ],
)
def test_a_malformed_embeddings_answer_is_an_error_not_a_vector(
    adapter: OpenAIUpstreamAdapter, fake_openai: FakeOpenAI, body: Response, reason: str
) -> None:
    fake_openai.queued.append(body)

    with pytest.raises(RemoteUpstreamError, match=reason):
        adapter.encode([Item(text="a")], ["dense"])


@pytest.mark.parametrize(
    ("body", "reason"),
    [
        (answer('{"results": [{"index": 0, "relevance_score": 0.5}]}'), "different number of rerank results"),
        (
            answer('{"results": [{"index": 0, "relevance_score": 0.5}, {"index": 0, "relevance_score": 0.4}]}'),
            "repeated index",
        ),
        (
            answer('{"results": [{"index": 0, "relevance_score": 0.5}, {"index": 2, "relevance_score": 0.4}]}'),
            "repeated index",
        ),
        (
            answer('{"results": [{"index": 0, "relevance_score": "0.5"}, {"index": 1, "relevance_score": 0.4}]}'),
            "not a finite number",
        ),
        (
            answer('{"results": [{"index": 0, "relevance_score": true}, {"index": 1, "relevance_score": 0.4}]}'),
            "not a finite number",
        ),
        (
            answer('{"results": [{"index": 0, "relevance_score": Infinity}, {"index": 1, "relevance_score": 0.4}]}'),
            "not a finite number",
        ),
    ],
    ids=["short", "repeated-index", "index-out-of-range", "string-score", "boolean-score", "infinite-score"],
)
def test_a_malformed_rerank_answer_is_an_error_not_a_score(
    adapter: OpenAIUpstreamAdapter, fake_openai: FakeOpenAI, body: Response, reason: str
) -> None:
    fake_openai.queued.append(body)
    query = Item(text="q")

    with pytest.raises(RemoteUpstreamError, match=reason):
        adapter.score_pairs([query, query], [Item(text="a"), Item(text="b")])


def test_extraction_is_refused(adapter: OpenAIUpstreamAdapter, fake_openai: FakeOpenAI) -> None:
    with pytest.raises(InvalidInputError, match="no extraction endpoint"):
        adapter.extract([Item(text="Alice met Bob")], labels=["person"])

    assert fake_openai.calls == []


def test_a_profile_on_an_upstream_of_another_kind_does_not_load(fake_openai: FakeOpenAI) -> None:
    install_upstreams(
        {"open-host": Upstream.model_validate({"kind": "sie", "base_url": fake_openai.url, "rate_cap": RATE_CAP})}
    )
    try:
        with pytest.raises(UpstreamConfigError, match="is not of kind 'openai'"):
            OpenAIUpstreamAdapter(upstream="open-host", upstream_model=UPSTREAM_MODEL).load("cpu")
    finally:
        install_upstreams({})


def test_an_unloaded_adapter_sends_nothing(adapter: OpenAIUpstreamAdapter, fake_openai: FakeOpenAI) -> None:
    adapter.unload()

    with pytest.raises(RuntimeError):
        adapter.encode([Item(text="a")], ["dense"])

    assert fake_openai.calls == []
