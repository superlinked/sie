"""Native encode/score/extract bodies are bounded before they are buffered.

Fixture mirrors ``test_error_code_hygiene.py``: a mocked registry on the
direct adapter path, with an adapter that raises a distinctive error so a
body that was read and parsed is observable as that error's status.
"""

from __future__ import annotations

from collections.abc import Iterator
from unittest.mock import MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sie_server.api import helpers
from sie_server.api.encode import router as encode_router
from sie_server.config.model import EmbeddingDim, EncodeTask, ModelConfig, ProfileConfig, Tasks
from sie_server.core.registry import ModelRegistry
from sie_server.core.worker import QueueFullError

JSON_HEADERS = {"Accept": "application/json", "Content-Type": "application/json"}


def _client() -> tuple[TestClient, MagicMock]:
    adapter = MagicMock()
    adapter.encode = MagicMock(side_effect=QueueFullError("queue occupied"))
    config = ModelConfig(
        sie_id="test-model",
        hf_id="org/test",
        tasks=Tasks(encode=EncodeTask(dense=EmbeddingDim(dim=3))),
        profiles={"default": ProfileConfig(adapter_path="test:TestAdapter", max_batch_tokens=8192)},
    )
    registry = MagicMock(spec=ModelRegistry)
    registry.has_model.return_value = True
    registry.is_loaded.return_value = True
    registry.is_loading.return_value = False
    registry.is_unloading.return_value = False
    registry.is_failed.return_value = False
    registry.get_failure.return_value = None
    registry.get.return_value = adapter
    registry.get_config.return_value = config
    registry.model_names = [config.sie_id]
    registry.device = "cpu"
    registry.engine_config = None
    preprocessor_registry = MagicMock()
    preprocessor_registry.has_tokenizer.return_value = False
    preprocessor_registry.has_preprocessor.return_value = False
    registry.preprocessor_registry = preprocessor_registry

    app = FastAPI()
    app.include_router(encode_router)
    app.state.registry = registry
    return TestClient(app), adapter


def _body(n_items: int) -> bytes:
    items = ",".join(f'{{"text": "item {i} padded out a little"}}' for i in range(n_items))
    return f'{{"items": [{items}]}}'.encode()


@pytest.fixture
def small_limit(monkeypatch: pytest.MonkeyPatch) -> int:
    limit = 256
    monkeypatch.setattr(helpers, "MAX_REQUEST_BODY_BYTES", limit)
    return limit


def test_declared_oversized_body_is_refused_before_reading(small_limit: int) -> None:
    client, adapter = _client()
    body = _body(64)
    assert len(body) > small_limit
    resp = client.post("/v1/encode/test-model", content=body, headers=JSON_HEADERS)
    assert resp.status_code == 413
    assert resp.json()["detail"]["code"] == "INPUT_TOO_LONG"
    adapter.encode.assert_not_called()


def test_chunked_body_is_refused_at_the_limit(small_limit: int) -> None:
    """No Content-Length: the bound has to hold on the stream itself."""
    client, adapter = _client()
    body = _body(64)

    def chunks() -> Iterator[bytes]:
        for start in range(0, len(body), 100):
            yield body[start : start + 100]

    resp = client.post("/v1/encode/test-model", content=chunks(), headers=JSON_HEADERS)
    assert resp.status_code == 413
    assert resp.json()["detail"]["code"] == "INPUT_TOO_LONG"
    adapter.encode.assert_not_called()


def test_body_within_the_limit_reaches_the_handler(small_limit: int) -> None:
    client, adapter = _client()
    body = _body(2)
    assert len(body) <= small_limit
    resp = client.post("/v1/encode/test-model", content=body, headers=JSON_HEADERS)
    # The adapter's QueueFullError is the proof the body was read and parsed.
    assert resp.status_code == 503
    assert resp.json()["detail"]["code"] == "QUEUE_FULL"
    adapter.encode.assert_called_once()


def test_default_limit_matches_the_gateway_native_ceiling(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("SIE_MAX_REQUEST_BODY_BYTES", raising=False)
    assert helpers.MAX_REQUEST_BODY_BYTES == 34 * 1024 * 1024


def test_negative_limit_is_rejected_at_startup(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SIE_MAX_REQUEST_BODY_BYTES", "-1")
    with pytest.raises(ValueError, match="non-negative"):
        helpers._request_body_limit_from_env()
    monkeypatch.setenv("SIE_MAX_REQUEST_BODY_BYTES", "0")
    assert helpers._request_body_limit_from_env() == 0
