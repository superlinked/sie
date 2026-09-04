# Contract tests for the usability-sweep error-code hygiene fixes.
#
# 1. ``InferenceErrorHandler.handle_queue_full`` — the old mapping returned
#    ``503 MODEL_NOT_LOADED`` for every ``QueueFullError``, which was wrong on
#    every axis: the model IS loaded, a request larger than the queue's total
#    capacity can never succeed no matter how often it is retried, and genuine
#    transient pressure carried no ``Retry-After`` hint. New contract:
#
#    - request item count alone exceeds the queue limit → ``400 INVALID_INPUT``
#      naming the per-request limit (permanent client error, no Retry-After);
#    - queue occupied by other work → ``503 QUEUE_FULL`` + ``Retry-After``;
#    - a legacy ``QueueFullError`` without structured counts is conservatively
#      treated as transient.
#
# 2. encode/score ``InputTooLongError`` — ``InputTooLongError`` subclasses
#    ``ValueError``, and encode/score lacked the dedicated except arm that
#    extract/audio already had, so token-budget overruns degraded to the
#    generic ``400 INVALID_INPUT`` instead of ``400 INPUT_TOO_LONG`` (which
#    the SDKs key their short-circuit behaviour on).
#
# Fixture patterns mirror ``test_extract_oom.py`` (failing worker future) and
# ``test_encode_endpoint.py`` (direct adapter path).

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import msgpack_numpy as m
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sie_server.adapters.base import ModelCapabilities, ModelDims
from sie_server.adapters.errors import InputTooLongError
from sie_server.api.encode import router as encode_router
from sie_server.api.score import router as score_router
from sie_server.config.model import (
    EmbeddingDim,
    EncodeTask,
    ModelConfig,
    ProfileConfig,
    ScoreTask,
    Tasks,
)
from sie_server.core.registry import ModelRegistry
from sie_server.core.worker import QueueFullError

m.patch()  # numpy <-> msgpack

JSON_HEADERS = {"Accept": "application/json"}


def _base_registry(config: ModelConfig, adapter: MagicMock) -> MagicMock:
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
    # ``oom_retry_after_from_registry`` falls back to the module default
    # when engine_config is None (see test_extract_oom.py).
    registry.engine_config = None
    return registry


def _build_encode_client(failure: BaseException) -> TestClient:
    """Encode client whose direct adapter call raises ``failure``.

    Uses the no-tokenizer direct path (as in ``test_encode_endpoint.py``)
    so the exception surfaces through the endpoint's except arms without
    a worker in the loop.
    """
    adapter = MagicMock()
    adapter.encode = MagicMock(side_effect=failure)
    config = ModelConfig(
        sie_id="test-model",
        hf_id="org/test",
        tasks=Tasks(encode=EncodeTask(dense=EmbeddingDim(dim=3))),
        profiles={"default": ProfileConfig(adapter_path="test:TestAdapter", max_batch_tokens=8192)},
    )
    registry = _base_registry(config, adapter)
    preprocessor_registry = MagicMock()
    preprocessor_registry.has_tokenizer.return_value = False
    preprocessor_registry.has_preprocessor.return_value = False
    registry.preprocessor_registry = preprocessor_registry

    app = FastAPI()
    app.include_router(encode_router)
    app.state.registry = registry
    return TestClient(app)


def _build_score_client(failure: BaseException) -> TestClient:
    """Score client whose worker future fails with ``failure``."""
    adapter = MagicMock()
    adapter.capabilities = ModelCapabilities(inputs=["text"], outputs=[])
    adapter.dims = ModelDims()

    worker = MagicMock()

    async def _failing_submit(*_args: Any, **_kwargs: Any) -> asyncio.Future[Any]:
        loop = asyncio.get_running_loop()
        future: asyncio.Future[Any] = loop.create_future()
        future.set_exception(failure)
        return future

    worker.submit_score = _failing_submit

    config = ModelConfig(
        sie_id="test-reranker",
        hf_id="org/test-reranker",
        tasks=Tasks(encode=EncodeTask(), score=ScoreTask()),
        profiles={"default": ProfileConfig(adapter_path="test:TestCrossEncoderAdapter", max_batch_tokens=8192)},
    )
    registry = _base_registry(config, adapter)
    registry.start_worker = AsyncMock(return_value=worker)

    app = FastAPI()
    app.include_router(score_router)
    app.state.registry = registry
    return TestClient(app)


def _post_encode(client: TestClient) -> Any:
    return client.post(
        "/v1/encode/test-model",
        json={"items": [{"text": "hello"}]},
        headers=JSON_HEADERS,
    )


def _post_score(client: TestClient) -> Any:
    return client.post(
        "/v1/score/test-reranker",
        json={"query": {"text": "q"}, "items": [{"text": "doc"}]},
        headers=JSON_HEADERS,
    )


def _detail(response: Any) -> dict[str, Any]:
    body = response.json()
    return body.get("detail", body)  # FastAPI wraps HTTPException in 'detail'


class TestQueueFullMapping:
    """handle_queue_full: 400 for can-never-fit, 503 QUEUE_FULL for pressure."""

    def test_over_capacity_request_maps_to_400_invalid_input(self) -> None:
        """A request bigger than the whole queue is a permanent client error."""
        failure = QueueFullError(
            "Queue full: 0 items pending, cannot add 1000 more (limit: 512)",
            pending=0,
            requested=1000,
            limit=512,
        )
        response = _post_encode(_build_encode_client(failure))

        assert response.status_code == 400, response.text
        # Never invite a retry for a request that can never fit.
        assert "Retry-After" not in response.headers
        detail = _detail(response)
        assert detail["code"] == "INVALID_INPUT"
        # The message names the per-request limit and the offending count.
        assert "512" in detail["message"]
        assert "1000" in detail["message"]

    def test_transient_pressure_maps_to_503_queue_full_with_retry_after(self) -> None:
        """Queue occupied by other work: retryable, and says so."""
        failure = QueueFullError(
            "Queue full: 500 items pending, cannot add 100 more (limit: 512)",
            pending=500,
            requested=100,
            limit=512,
        )
        response = _post_encode(_build_encode_client(failure))

        assert response.status_code == 503, response.text
        assert response.headers.get("Retry-After") == "1"
        detail = _detail(response)
        # Regression guard: was MODEL_NOT_LOADED, which is false (the model
        # IS loaded) and not in any SDK retryable set.
        assert detail["code"] == "QUEUE_FULL"
        assert "Queue full" in detail["message"]

    def test_legacy_queue_full_without_counts_is_treated_as_transient(self) -> None:
        """A message-only QueueFullError conservatively maps to 503 QUEUE_FULL."""
        response = _post_encode(_build_encode_client(QueueFullError("Queue full")))

        assert response.status_code == 503, response.text
        assert response.headers.get("Retry-After") == "1"
        assert _detail(response)["code"] == "QUEUE_FULL"

    def test_score_transient_pressure_maps_to_503_queue_full(self) -> None:
        """The score endpoint shares the same queue-full contract."""
        failure = QueueFullError(
            "Queue full: 512 items pending, cannot add 1 more (limit: 512)",
            pending=512,
            requested=1,
            limit=512,
        )
        response = _post_score(_build_score_client(failure))

        assert response.status_code == 503, response.text
        assert response.headers.get("Retry-After") == "1"
        assert _detail(response)["code"] == "QUEUE_FULL"

    def test_score_over_capacity_request_maps_to_400(self) -> None:
        failure = QueueFullError(
            "Queue full: 0 items pending, cannot add 600 more (limit: 512)",
            pending=0,
            requested=600,
            limit=512,
        )
        response = _post_score(_build_score_client(failure))

        assert response.status_code == 400, response.text
        detail = _detail(response)
        assert detail["code"] == "INVALID_INPUT"
        assert "512" in detail["message"]


class TestInputTooLongMapping:
    """encode/score must map InputTooLongError to INPUT_TOO_LONG, not INVALID_INPUT."""

    def test_encode_input_too_long_maps_to_input_too_long_code(self) -> None:
        response = _post_encode(_build_encode_client(InputTooLongError("input of 9000 tokens exceeds budget")))

        assert response.status_code == 400, response.text
        detail = _detail(response)
        # Regression guard: InputTooLongError subclasses ValueError, and
        # without the dedicated arm it degraded to INVALID_INPUT.
        assert detail["code"] == "INPUT_TOO_LONG"
        assert "9000" in detail["message"]

    def test_score_input_too_long_maps_to_input_too_long_code(self) -> None:
        response = _post_score(_build_score_client(InputTooLongError("pair of 9000 tokens exceeds budget")))

        assert response.status_code == 400, response.text
        assert _detail(response)["code"] == "INPUT_TOO_LONG"

    def test_encode_plain_value_error_still_maps_to_invalid_input(self) -> None:
        """The generic ValueError arm keeps its INVALID_INPUT mapping."""
        response = _post_encode(_build_encode_client(ValueError("bad input")))

        assert response.status_code == 400, response.text
        assert _detail(response)["code"] == "INVALID_INPUT"

    def test_score_plain_value_error_still_maps_to_invalid_input(self) -> None:
        response = _post_score(_build_score_client(ValueError("bad input")))

        assert response.status_code == 400, response.text
        assert _detail(response)["code"] == "INVALID_INPUT"


class TestUnexpectedInferenceErrorSanitization:
    @pytest.mark.parametrize(
        ("post", "build_client", "message"),
        [
            (_post_encode, _build_encode_client, "internal error during encoding"),
            (_post_score, _build_score_client, "internal error during scoring"),
        ],
        ids=["encode", "score"],
    )
    def test_internal_exception_text_is_not_returned(
        self,
        post: Any,
        build_client: Any,
        message: str,
    ) -> None:
        response = post(build_client(RuntimeError("sensitive worker detail")))

        assert response.status_code == 500
        detail = _detail(response)
        assert detail == {"code": "INFERENCE_ERROR", "message": message}
        assert "sensitive worker detail" not in response.text
