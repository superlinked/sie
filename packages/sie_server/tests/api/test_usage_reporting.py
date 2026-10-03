"""What `usage` reports on /v1/encode and /v1/extract.

The reported number must be the worker's own post-tokenization count — the same
basis telemetry meters from — never a character estimate, and a measured zero
must be reported as `0` rather than read as "unknown".
"""

import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sie_server.api.encode import router as encode_router
from sie_server.api.extract import router as extract_router
from sie_server.api.helpers import validated_total
from sie_server.config.model import (
    EmbeddingDim,
    EncodeTask,
    ExtractTask,
    ModelConfig,
    ProfileConfig,
    Tasks,
)
from sie_server.core.inference_output import EncodeOutput, ExtractOutput
from sie_server.core.postprocessor_registry import PostprocessorRegistry
from sie_server.core.registry import ModelRegistry
from sie_server.core.timing import RequestTiming
from sie_server.core.worker import WorkerResult

JSON_HEADERS = {"Accept": "application/json"}

# 48 characters. The OpenAI-compat layer's character estimate for this input is
# 48 / 4 = 12, so a reported 12 would prove the estimate leaked through.
LONG_TEXT = "x" * 48
WORKER_TOKEN_COUNT = 7


def _encode_registry(extra: dict[str, Any]) -> MagicMock:
    """Registry whose adapter stamps `extra` on its EncodeOutput.

    `extra["input_token_counts"]` is the authoritative per-item basis the
    flash adapters expose; the encode pipeline lifts it onto the request
    timing, which is where both telemetry and `usage` read it from.
    """

    def encode_impl(items: list[Any], output_types: list[str], **kwargs: Any) -> EncodeOutput:
        output = EncodeOutput(
            dense=np.array([[0.1, 0.2, 0.3]] * len(items), dtype=np.float32),
            batch_size=len(items),
            dense_dim=3,
        )
        output.extra.update(extra)
        return output

    adapter = MagicMock()
    adapter.encode = MagicMock(side_effect=encode_impl)
    # A real adapter's `count_input_tokens` is the pipeline's next fallback;
    # returning None keeps these tests on the `extra` basis they are about.
    adapter.count_input_tokens = MagicMock(return_value=None)

    registry = MagicMock(spec=ModelRegistry)
    registry.has_model.return_value = True
    registry.is_loaded.return_value = True
    registry.is_loading.return_value = False
    registry.is_unloading.return_value = False
    registry.is_failed.return_value = False
    registry.get_failure.return_value = None
    registry.get.return_value = adapter
    registry.get_config.return_value = ModelConfig(
        sie_id="test-model",
        hf_id="org/test",
        tasks=Tasks(encode=EncodeTask(dense=EmbeddingDim(dim=3))),
        profiles={"default": ProfileConfig(adapter_path="test:TestAdapter", max_batch_tokens=8192)},
    )
    registry.model_names = ["test-model"]
    registry.device = "cpu"
    preprocessor_registry = MagicMock()
    preprocessor_registry.has_tokenizer.return_value = False
    preprocessor_registry.has_preprocessor.return_value = False
    registry.preprocessor_registry = preprocessor_registry
    registry.postprocessor_registry = PostprocessorRegistry(ThreadPoolExecutor(max_workers=1))
    return registry


def _encode_client(extra: dict[str, Any]) -> TestClient:
    app = FastAPI()
    app.include_router(encode_router)
    app.state.registry = _encode_registry(extra)
    return TestClient(app)


def _encode(client: TestClient, items: list[dict[str, Any]]) -> dict[str, Any]:
    response = client.post("/v1/encode/test-model", json={"items": items}, headers=JSON_HEADERS)
    assert response.status_code == 200, response.text
    return response.json()


class TestEncodeUsage:
    def test_reports_the_worker_count_not_a_character_estimate(self) -> None:
        body = _encode(
            _encode_client({"input_token_counts": [WORKER_TOKEN_COUNT]}),
            [{"text": LONG_TEXT}],
        )
        assert body["usage"]["input_tokens"] == WORKER_TOKEN_COUNT
        assert body["usage"]["input_tokens"] != len(LONG_TEXT) // 4

    def test_sums_the_worker_counts_across_the_batch(self) -> None:
        body = _encode(
            _encode_client({"input_token_counts": [3, 11]}),
            [{"text": "first"}, {"text": "second"}],
        )
        assert body["usage"]["input_tokens"] == 14

    def test_reports_an_authoritative_zero(self) -> None:
        """A media-only encode reads no text: `0` is the measurement.

        The metering seam deliberately drops that zero (it is only meaningful
        there under an image witness), but a caller reading `usage` must be able
        to tell "measured nothing" from "could not measure".
        """
        body = _encode(
            _encode_client({"input_token_counts": [0], "input_image_counts": [4]}),
            [{"image": {"url": "https://example.invalid/cat.png"}}],
        )
        assert body["usage"]["input_tokens"] == 0
        assert body["usage"]["images"] == 4

    def test_omits_usage_when_no_authoritative_count_exists(self) -> None:
        """Omission, not an estimate. An absent block means "unknown"."""
        body = _encode(_encode_client({}), [{"text": LONG_TEXT}])
        assert "usage" not in body

    def test_omits_usage_when_the_counts_are_misaligned_with_the_batch(self) -> None:
        body = _encode(
            _encode_client({"input_token_counts": [3]}),
            [{"text": "first"}, {"text": "second"}],
        )
        assert "usage" not in body


def _extract_client(input_token_counts: list[int] | None) -> TestClient:
    output = ExtractOutput(entities=[[]], batch_size=1, input_token_counts=input_token_counts)
    timing = RequestTiming()
    timing.start_tokenization()
    timing.end_tokenization()
    timing.start_queue()
    timing.start_inference()
    timing.end_inference()
    timing.finish()

    async def submit_extract(prepared_items: Any, items: Any, **kwargs: Any) -> Any:
        future: asyncio.Future[WorkerResult] = asyncio.get_running_loop().create_future()
        future.set_result(WorkerResult(output=output, timing=kwargs.get("timing") or timing))
        return future

    worker = MagicMock()
    worker.submit_extract = submit_extract

    registry = MagicMock(spec=ModelRegistry)
    registry.has_model.return_value = True
    registry.is_loaded.return_value = True
    registry.is_loading.return_value = False
    registry.is_unloading.return_value = False
    registry.is_failed.return_value = False
    registry.get_failure.return_value = None
    registry.get.return_value = MagicMock()
    registry.get_config.return_value = ModelConfig(
        sie_id="test-model",
        hf_id="org/test",
        tasks=Tasks(extract=ExtractTask()),
        profiles={"default": ProfileConfig(adapter_path="test:TestAdapter", max_batch_tokens=8192)},
    )
    registry.model_names = ["test-model"]
    registry.device = "cpu"
    registry.start_worker = AsyncMock(return_value=worker)

    app = FastAPI()
    app.include_router(extract_router)
    app.state.registry = registry
    return TestClient(app)


def _extract(client: TestClient) -> dict[str, Any]:
    response = client.post(
        "/v1/extract/test-model",
        json={"items": [{"text": LONG_TEXT}], "params": {"labels": ["person"]}},
        headers=JSON_HEADERS,
    )
    assert response.status_code == 200, response.text
    return response.json()


class TestExtractUsage:
    def test_reports_the_worker_count(self) -> None:
        body = _extract(_extract_client([WORKER_TOKEN_COUNT]))
        assert body["usage"]["input_tokens"] == WORKER_TOKEN_COUNT

    def test_reports_an_authoritative_zero(self) -> None:
        body = _extract(_extract_client([0]))
        assert body["usage"]["input_tokens"] == 0

    def test_omits_usage_when_no_authoritative_count_exists(self) -> None:
        body = _extract(_extract_client(None))
        assert "usage" not in body


@pytest.mark.parametrize(
    ("counts", "item_count", "expected"),
    [
        ([5, 6], 2, 11),
        ([0, 0], 2, 0),
        (None, 2, None),
        ([5], 2, None),
        ([True, 1], 2, None),
    ],
)
def test_validated_total_keeps_zero_and_drops_the_unusable(
    counts: list[int] | None, item_count: int, expected: int | None
) -> None:
    assert validated_total(counts, item_count) == expected
