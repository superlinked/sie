# Tests for the extract endpoint's OOM → 503 RESOURCE_EXHAUSTED mapping.
#
# This is a contract test for ``InferenceErrorHandler.handle_inference_error``:
# when the worker raises an OOM (or its ``ResourceExhaustedError`` wrapper),
# the API must return HTTP 503 with the ``Retry-After`` header and a
# ``RESOURCE_EXHAUSTED`` error code so the SDK can auto-retry.
#
# We follow the same fixture pattern as ``test_extract.py`` but inject a
# worker whose ``submit_extract`` produces a future that fails with an OOM
# exception.

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import msgpack_numpy as m
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sie_server.adapters.base import ModelCapabilities, ModelDims
from sie_server.api.extract import router as extract_router
from sie_server.config.engine import EngineConfig
from sie_server.config.model import (
    EmbeddingDim,  # noqa: F401 — imported for parity with sibling test module
    EncodeTask,
    ExtractTask,
    ModelConfig,
    ProfileConfig,
    Tasks,
)
from sie_server.core.oom import ResourceExhausted, ResourceExhaustedError
from sie_server.core.registry import ModelRegistry

m.patch()  # numpy <-> msgpack

JSON_HEADERS = {"Accept": "application/json"}


def _build_client(failure: BaseException) -> TestClient:
    """Wire up a TestClient whose extract worker future fails with ``failure``."""
    adapter = MagicMock()
    adapter.capabilities = ModelCapabilities(inputs=["text"], outputs=[])
    adapter.dims = ModelDims()

    worker = MagicMock()

    async def _failing_submit(*_args: Any, **_kwargs: Any) -> asyncio.Future[Any]:
        loop = asyncio.get_running_loop()
        future: asyncio.Future[Any] = loop.create_future()
        future.set_exception(failure)
        return future

    worker.submit_extract = _failing_submit

    registry = MagicMock(spec=ModelRegistry)
    registry.has_model.return_value = True
    registry.is_loaded.return_value = True
    registry.is_loading.return_value = False
    registry.is_unloading.return_value = False
    registry.get.return_value = adapter
    registry.get_config.return_value = ModelConfig(
        sie_id="test-extractor",
        hf_id="org/test-extractor",
        tasks=Tasks(encode=EncodeTask(), extract=ExtractTask()),
        profiles={"default": ProfileConfig(adapter_path="test:Adapter", max_batch_tokens=8192)},
    )
    registry.model_names = ["test-extractor"]
    registry.device = "cpu"
    registry.start_worker = AsyncMock(return_value=worker)
    # ``oom_retry_after_from_registry`` reads ``registry.engine_config``;
    # ``MagicMock(spec=ModelRegistry)`` returns a Mock for it which would
    # blow up the ``Retry-After`` header. Set None to fall through to
    # the module default constant (5).
    registry.engine_config = None

    app = FastAPI()
    app.include_router(extract_router)
    app.state.registry = registry
    return TestClient(app)


@pytest.mark.parametrize(
    "failure",
    [
        # Plain OOM (recovery disabled or escaped): InferenceErrorHandler
        # must still recognise it via is_oom_error and map to 503.
        RuntimeError("CUDA out of memory. Tried to allocate 2 GiB"),
        # Worker recovery exhausted — the wrapped error is the structural
        # signal but the substring matcher also catches the wrapped message.
        ResourceExhaustedError(
            "Resource exhausted: CUDA out of memory",
            marker=ResourceExhausted(operation="extract", attempts=4, original_message="CUDA out of memory"),
        ),
    ],
    ids=["raw-cuda-oom", "wrapped-resource-exhausted"],
)
def test_extract_oom_maps_to_503_resource_exhausted(failure: BaseException) -> None:
    client = _build_client(failure)

    response = client.post(
        "/v1/extract/test-extractor",
        json={"items": [{"text": "hello"}], "params": {"labels": ["x"]}},
        headers=JSON_HEADERS,
    )

    assert response.status_code == 503, response.text
    assert response.headers.get("Retry-After") == "5"

    body = response.json()
    detail = body.get("detail", body)  # FastAPI wraps in 'detail' for HTTPException
    assert detail["code"] == "RESOURCE_EXHAUSTED"
    # Message should mention the operation name, not leak internals
    assert "Extraction" in detail["message"] or "extract" in detail["message"].lower()


def test_extract_non_oom_keeps_500_inference_error() -> None:
    """Regression guard: a non-OOM exception still maps to 500 INFERENCE_ERROR."""
    client = _build_client(RuntimeError("model produced bad shape"))

    response = client.post(
        "/v1/extract/test-extractor",
        json={"items": [{"text": "hello"}], "params": {"labels": ["x"]}},
        headers=JSON_HEADERS,
    )

    assert response.status_code == 500
    assert "Retry-After" not in response.headers
    body = response.json()
    detail = body.get("detail", body)
    assert detail["code"] == "INFERENCE_ERROR"


def test_extract_oom_retry_after_honours_engine_config() -> None:
    """The Retry-After header reflects ``engine_config.oom_recovery.retry_after_s``.

    Regression: the header was previously hardcoded to 5 regardless of the
    operator-tunable engine config. Verifies that
    ``oom_retry_after_from_registry`` threads the value through correctly.
    """
    # Build an engine_config with retry_after_s=12 (within the 1-60 range).
    engine_config = EngineConfig.model_validate({"oom_recovery": {"retry_after_s": 12}})

    client = _build_client(RuntimeError("CUDA out of memory"))
    # Override the registry's engine_config that _build_client set to None.
    client.app.state.registry.engine_config = engine_config

    response = client.post(
        "/v1/extract/test-extractor",
        json={"items": [{"text": "hello"}], "params": {"labels": ["x"]}},
        headers=JSON_HEADERS,
    )

    assert response.status_code == 503
    assert response.headers.get("Retry-After") == "12"


def test_extract_oom_with_recovery_disabled_still_maps_to_503() -> None:
    """Kill switch disables in-worker recovery, NOT the OOM 503 mapping.

    When ``SIE_DISABLE_OOM_RECOVERY=1`` (or
    ``oom_recovery.enabled=False``), ``BatchExecutor`` re-raises OOMs
    unchanged. The HTTP layer's ``InferenceErrorHandler`` still detects
    the OOM via ``is_oom_error`` and returns 503 RESOURCE_EXHAUSTED — so
    SDK auto-retry still works even with recovery off.

    Regression guard for an operator turning off recovery during an
    incident and accidentally also turning off the SDK retry contract.
    """
    engine_config = EngineConfig.model_validate({"oom_recovery": {"enabled": False}})
    assert engine_config.oom_recovery.enabled is False

    client = _build_client(RuntimeError("CUDA out of memory"))
    client.app.state.registry.engine_config = engine_config

    response = client.post(
        "/v1/extract/test-extractor",
        json={"items": [{"text": "hello"}], "params": {"labels": ["x"]}},
        headers=JSON_HEADERS,
    )

    # Critical: still 503, not 500.
    assert response.status_code == 503
    detail = response.json().get("detail", response.json())
    assert detail["code"] == "RESOURCE_EXHAUSTED"
