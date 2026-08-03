"""Input-contract tests for the Mac-local OpenAI routes (/v1/chat/completions, /v1/rerank).

These run on any platform — they exercise the request validation that happens
BEFORE the registry/model is touched, so no MLX subprocess or torch is needed.
The happy paths are covered live by ``mise run mac-smoke`` (Apple Silicon).
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sie_server.api.openai_local import _validate_mlx_seed, router


def _client() -> TestClient:
    app = FastAPI()
    app.include_router(router)
    # Registry is only reached AFTER input validation, so a stub is fine for 4xx paths.
    app.state.registry = MagicMock()
    return TestClient(app, raise_server_exceptions=False)


# -- /v1/chat/completions validation -----------------------------------------


def test_chat_requires_object_body() -> None:
    r = _client().post("/v1/chat/completions", content=b"[]", headers={"content-type": "application/json"})
    assert r.status_code == 400


def test_chat_requires_model() -> None:
    r = _client().post("/v1/chat/completions", json={"messages": [{"role": "user", "content": "hi"}]})
    assert r.status_code == 400
    assert r.json()["detail"]["param"] == "model"


def test_chat_requires_nonempty_messages() -> None:
    r = _client().post("/v1/chat/completions", json={"model": "m", "messages": []})
    assert r.status_code == 400
    assert r.json()["detail"]["param"] == "messages"


def test_chat_rejects_non_generation_model_before_load() -> None:
    # An embedding/reranker model (no generate task) must be rejected BEFORE ensure_loaded()
    # so a chat request can't kick off a real model load that only 501s afterwards.
    client = _client()
    client.app.state.registry.get_config.return_value.tasks.generate = None
    r = client.post(
        "/v1/chat/completions", json={"model": "embed-model", "messages": [{"role": "user", "content": "hi"}]}
    )
    assert r.status_code == 400
    assert "generation" in r.json()["detail"]["message"].lower()


def test_chat_rejects_non_bool_stream() -> None:
    msgs = [{"role": "user", "content": "hi"}]
    r = _client().post("/v1/chat/completions", json={"model": "m", "messages": msgs, "stream": "false"})
    assert r.status_code == 400
    assert r.json()["detail"]["param"] == "stream"


def test_chat_rejects_invalid_max_tokens() -> None:
    msgs = [{"role": "user", "content": "hi"}]
    for bad in (0, -5, "100", True):
        r = _client().post("/v1/chat/completions", json={"model": "m", "messages": msgs, "max_tokens": bad})
        assert r.status_code == 400, bad
        assert r.json()["detail"]["param"] == "max_tokens"


@pytest.mark.parametrize(
    ("bad", "message"),
    [
        (True, "'seed' must be an integer"),
        ("1", "'seed' must be an integer"),
        (1.5, "'seed' must be an integer"),
        (-(1 << 63) - 1, "'seed' is outside the supported integer range"),
        (1 << 63, "'seed' is outside the supported integer range"),
    ],
)
def test_chat_rejects_invalid_seed(bad: object, message: str) -> None:
    msgs = [{"role": "user", "content": "hi"}]
    r = _client().post("/v1/chat/completions", json={"model": "m", "messages": msgs, "seed": bad})
    assert r.status_code == 400, bad
    assert r.json()["detail"] == {
        "code": "INVALID_INPUT",
        "message": message,
        "param": "seed",
    }


@pytest.mark.parametrize(
    ("seed", "expected"),
    [
        (-(1 << 63), 1 << 63),
        (-1, (1 << 64) - 1),
        (0, 0),
        ((1 << 63) - 1, (1 << 63) - 1),
    ],
)
def test_validate_mlx_seed_preserves_signed_bit_pattern(seed: int, expected: int) -> None:
    assert _validate_mlx_seed(seed) == expected


def test_validate_mlx_seed_preserves_absent_value() -> None:
    assert _validate_mlx_seed(None) is None


# -- /v1/rerank validation ----------------------------------------------------


def test_rerank_requires_model() -> None:
    r = _client().post("/v1/rerank", json={"query": "q", "documents": ["a"]})
    assert r.status_code == 400
    assert r.json()["detail"]["param"] == "model"


def test_rerank_requires_query() -> None:
    r = _client().post("/v1/rerank", json={"model": "m", "documents": ["a"]})
    assert r.status_code == 400
    assert r.json()["detail"]["param"] == "query"


def test_rerank_rejects_blank_model_and_query() -> None:
    for body, param in [
        ({"model": "   ", "query": "q", "documents": ["a"]}, "model"),
        ({"model": "m", "query": "   ", "documents": ["a"]}, "query"),
    ]:
        r = _client().post("/v1/rerank", json=body)
        assert r.status_code == 400
        assert r.json()["detail"]["param"] == param


def test_rerank_requires_nonempty_string_documents() -> None:
    for docs in ([], "not-a-list", [1, 2], ["ok", 3], ["   "]):
        r = _client().post("/v1/rerank", json={"model": "m", "query": "q", "documents": docs})
        assert r.status_code == 400, docs
        assert r.json()["detail"]["param"] == "documents"


def test_rerank_top_n_must_be_positive_int() -> None:
    for bad in (0, -1, "3", True):
        r = _client().post("/v1/rerank", json={"model": "m", "query": "q", "documents": ["a"], "top_n": bad})
        assert r.status_code == 400, bad
        assert r.json()["detail"]["param"] == "top_n"


def test_rerank_rejects_too_many_documents() -> None:
    from sie_server.api import openai_local

    docs = ["d"] * (openai_local._MAX_RERANK_DOCS + 1)
    r = _client().post("/v1/rerank", json={"model": "m", "query": "q", "documents": docs})
    assert r.status_code == 400
    assert r.json()["detail"]["param"] == "documents"


def test_rerank_rejects_non_bool_return_documents() -> None:
    r = _client().post("/v1/rerank", json={"model": "m", "query": "q", "documents": ["a"], "return_documents": "true"})
    assert r.status_code == 400
    assert r.json()["detail"]["param"] == "return_documents"


def test_rerank_rejects_unknown_fields() -> None:
    r = _client().post("/v1/rerank", json={"model": "m", "query": "q", "documents": ["a"], "priority": 1})
    assert r.status_code == 400
    assert r.json()["detail"]["param"] == "priority"
