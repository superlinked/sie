from __future__ import annotations

import time
from unittest.mock import MagicMock

import pytest
from sie_sdk.client._shared import (
    handle_error,
    next_stream_retry_delay,
    raise_if_estimate_unroutable,
    raise_if_model_load_failed,
    sse_chunk_error,
)
from sie_sdk.client.errors import (
    AccountInactiveError,
    AccountStateUnavailableError,
    EstimateUnroutableError,
    InputTooLongError,
    InsufficientCreditsError,
    ModelLoadFailedError,
    ProvisioningError,
    RateLimitError,
    ResourceExhaustedError,
    ServerError,
    SpendLimitError,
)


def _response(status_code: int, code: str, *, param: str | None = "top_k") -> MagicMock:
    response = MagicMock()
    response.status_code = status_code
    response.headers = {"content-type": "application/json"}
    response.json.return_value = {
        "error": {
            "code": code,
            "message": "request failed",
            "param": param,
        }
    }
    response.text = ""
    return response


@pytest.mark.parametrize(
    ("status_code", "code", "error_type"),
    [
        (503, "PROVISIONING", ProvisioningError),
        (400, "INPUT_TOO_LONG", InputTooLongError),
        (429, "RATE_LIMIT", RateLimitError),
        (402, "INSUFFICIENT_CREDITS", InsufficientCreditsError),
        (402, "KEY_SPEND_LIMIT_EXCEEDED", SpendLimitError),
        (403, "ACCOUNT_SUSPENDED", AccountInactiveError),
        (503, "ACCOUNT_STATE_UNAVAILABLE", AccountStateUnavailableError),
    ],
)
def test_handle_error_preserves_param_on_specialized_errors(
    status_code: int,
    code: str,
    error_type: type[Exception],
) -> None:
    with pytest.raises(error_type) as exc_info:
        handle_error(_response(status_code, code))

    assert isinstance(
        exc_info.value,
        (
            ProvisioningError,
            InputTooLongError,
            RateLimitError,
            InsufficientCreditsError,
            SpendLimitError,
            AccountInactiveError,
            AccountStateUnavailableError,
        ),
    )
    assert exc_info.value.param == "top_k"


def test_model_load_failed_preserves_param() -> None:
    with pytest.raises(ModelLoadFailedError) as exc_info:
        raise_if_model_load_failed(_response(502, "MODEL_LOAD_FAILED", param="model"), model="m")

    assert exc_info.value.param == "model"


def test_estimate_unroutable_preserves_param() -> None:
    with pytest.raises(EstimateUnroutableError) as exc_info:
        raise_if_estimate_unroutable(_response(503, "QUEUE_UNAVAILABLE", param="model"))

    assert exc_info.value.param == "model"


def test_streaming_resource_exhausted_preserves_param() -> None:
    with pytest.raises(ResourceExhaustedError) as exc_info:
        next_stream_retry_delay(
            _response(503, "RESOURCE_EXHAUSTED", param="model"),
            model="m",
            gpu=None,
            wait_for_capacity=False,
            start_time=time.monotonic(),
            timeout=10.0,
            oom_retries=0,
            max_oom_retries=3,
        )

    assert exc_info.value.param == "model"


def test_streaming_gateway_timeout_preserves_param() -> None:
    with pytest.raises(ServerError) as exc_info:
        next_stream_retry_delay(
            _response(504, "GATEWAY_TIMEOUT", param="model"),
            model="m",
            gpu=None,
            wait_for_capacity=True,
            start_time=time.monotonic(),
            timeout=10.0,
            oom_retries=0,
            max_oom_retries=3,
        )

    assert exc_info.value.param == "model"


@pytest.mark.parametrize(
    ("code", "retry_after_s", "expected"),
    [
        pytest.param("RESOURCE_EXHAUSTED", 12, 12, id="valid"),
        pytest.param("RESOURCE_EXHAUSTED", None, None, id="null"),
        pytest.param("RESOURCE_EXHAUSTED", True, None, id="boolean"),
        pytest.param("RESOURCE_EXHAUSTED", 12.5, None, id="fractional"),
        pytest.param("RESOURCE_EXHAUSTED", "12", None, id="string"),
        pytest.param("RESOURCE_EXHAUSTED", 0, None, id="below-domain"),
        pytest.param("RESOURCE_EXHAUSTED", 61, None, id="above-domain"),
        pytest.param("MODEL_LOADING", 12, None, id="wrong-code"),
    ],
)
def test_sse_chunk_error_validates_retry_after_s(
    code: str,
    retry_after_s: object,
    expected: int | None,
) -> None:
    parsed = sse_chunk_error(
        {
            "error": {
                "code": code,
                "message": "request failed",
                "retry_after_s": retry_after_s,
            }
        }
    )

    assert parsed == (code, "request failed", None, expected)


def test_sse_chunk_error_tolerates_missing_retry_after_s() -> None:
    assert sse_chunk_error({"error": {"code": "RESOURCE_EXHAUSTED", "message": "full"}}) == (
        "RESOURCE_EXHAUSTED",
        "full",
        None,
        None,
    )
