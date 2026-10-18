"""OpenAI Decisions wire contract and the direct worker dispatch boundary."""

import copy
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient
from sie_server.api import decisions, systemone
from sie_server.config.model import ExtractTask, ModelConfig, ProfileConfig, Tasks
from sie_server.core.inference_output import ExtractItemError, ExtractOutput
from sie_server.core.registry import ModelRegistry
from sie_server.core.timing import RequestTiming
from sie_server.core.worker import WorkerResult
from sie_server.types.responses import ErrorCode

FIXTURE = json.loads((Path(__file__).resolve().parents[3] / "wire-fixtures" / "decisions.json").read_text())


def assert_contract(actual, expected):
    if isinstance(expected, dict):
        assert set(actual) == set(expected)
        for key, value in expected.items():
            assert_contract(actual[key], value)
    elif isinstance(expected, list):
        assert len(actual) == len(expected)
        for left, right in zip(actual, expected, strict=True):
            assert_contract(left, right)
    elif isinstance(expected, float):
        assert actual == pytest.approx(expected)
    else:
        assert type(actual) is type(expected)
        assert actual == expected


@pytest.mark.parametrize("case", FIXTURE["requests"], ids=lambda case: case["name"])
def test_shared_contract(case):
    request, questions = decisions.normalize_request(case["request"])
    assert request.model_dump() == case["systemone"]
    response = decisions.response_from_systemone(
        {"model": request.model, "answers": case["native_answers"], "usage": {"input_tokens": 42}}, questions
    )
    assert_contract(response, case["response"])


@pytest.mark.parametrize("case", FIXTURE["invalid_requests"], ids=lambda case: case["name"])
def test_shared_invalid_contract(case):
    with pytest.raises(HTTPException) as caught:
        decisions.normalize_request(case["request"])
    assert caught.value.status_code == 400
    assert isinstance(caught.value.detail, dict)
    assert caught.value.detail["param"] == case["param"]


@pytest.fixture
def dispatch(monkeypatch):
    case = FIXTURE["requests"][0]
    config = ModelConfig(
        sie_id=case["request"]["model"],
        hf_id=case["request"]["model"],
        tasks=Tasks(extract=ExtractTask()),
        profiles={
            "default": ProfileConfig(
                adapter_path="sie_server.adapters.gliner2.decide:GLiNER2DecideAdapter", max_batch_tokens=8192
            )
        },
    )
    registry = MagicMock(spec=ModelRegistry)
    registry.has_model.return_value = True
    registry.get_config.return_value = config
    route = SimpleNamespace(key="route", headers=lambda: {"X-SIE-Served-By": "local"})
    route_mock = AsyncMock(return_value=route)
    monkeypatch.setattr(systemone, "route_request", route_mock)
    timing = RequestTiming()
    output = ExtractOutput(entities=[[]], data=[copy.deepcopy(case["native_answers"])], input_token_counts=[42])
    worker = AsyncMock(return_value=WorkerResult(output=output, timing=timing))
    monkeypatch.setattr(systemone, "_extract_via_worker", worker)
    app = FastAPI()
    app.include_router(decisions.router)
    app.include_router(systemone.router)
    app.state.registry = registry
    return SimpleNamespace(client=TestClient(app), worker=worker, route=route_mock, registry=registry, output=output)


def test_route_uses_native_worker_and_preserves_headers(dispatch):
    case = FIXTURE["requests"][0]
    response = dispatch.client.post("/v1/decisions", json=case["request"])
    assert response.status_code == 200
    assert_contract(response.json(), case["response"])
    assert response.headers["x-sie-served-by"] == "local"
    assert int(response.headers["content-length"]) == len(response.content)
    args = dispatch.worker.call_args
    assert args.args[2][0].text == case["request"]["input"]
    assert args.kwargs["output_schema"] == case["systemone"]["questions"]
    assert args.kwargs["options"]["overflow_policy"] == "error"


@pytest.mark.parametrize("case", FIXTURE["invalid_requests"], ids=lambda case: case["name"])
def test_invalid_request_does_not_dispatch(dispatch, case):
    response = dispatch.client.post("/v1/decisions", json=case["request"])
    assert response.status_code == 400
    assert response.json()["error"]["param"] == case["param"]
    dispatch.worker.assert_not_awaited()
    dispatch.route.assert_not_awaited()


def test_unknown_model_error_is_openai_shaped(dispatch):
    dispatch.registry.has_model.return_value = False
    dispatch.registry.model_names = []
    response = dispatch.client.post("/v1/decisions", json=FIXTURE["requests"][0]["request"])
    assert response.status_code == 404
    assert response.json()["error"]["code"] == "MODEL_NOT_FOUND"
    dispatch.worker.assert_not_awaited()


def test_retry_headers_survive(dispatch):
    dispatch.route.side_effect = HTTPException(
        503, {"code": "MODEL_LOADING", "message": "loading"}, headers={"Retry-After": "30"}
    )
    response = dispatch.client.post("/v1/decisions", json=FIXTURE["requests"][0]["request"])
    assert response.status_code == 503
    assert response.headers["retry-after"] == "30"
    assert response.json()["error"]["code"] == "MODEL_LOADING"


@pytest.mark.parametrize("invalid_probability", [-0.1, 1.1, True, "0.9", None, float("nan"), float("inf")])
def test_invalid_predicate_output_is_server_error(dispatch, invalid_probability):
    dispatch.output.data[0]["question_0"]["noul"] = invalid_probability
    response = dispatch.client.post("/v1/decisions", json=FIXTURE["requests"][0]["request"])
    assert response.status_code == 500
    assert response.json()["error"]["type"] == "server_error"


@pytest.mark.parametrize(
    "probabilities", [{"option_0": 0.9}, {"option_0": 0.9, "option_1": 0.9}, {"option_0": -0.1, "option_1": 1.1}]
)
def test_invalid_distribution_is_server_error(dispatch, probabilities):
    dispatch.output.data[0]["question_1"]["probabilities"] = probabilities
    response = dispatch.client.post("/v1/decisions", json=FIXTURE["requests"][0]["request"])
    assert response.status_code == 500


def test_body_limit_precedes_model_work(dispatch, monkeypatch):
    monkeypatch.setattr(decisions, "MAX_BODY_BYTES", 16)
    response = dispatch.client.post("/v1/decisions", json=FIXTURE["requests"][0]["request"])
    assert response.status_code == 413
    assert "error" in response.json()
    dispatch.worker.assert_not_awaited()


@pytest.mark.parametrize("raw", ["{", "[]", "null"])
def test_malformed_body_is_openai_shaped(dispatch, raw):
    response = dispatch.client.post("/v1/decisions", content=raw)
    assert response.status_code == 400
    assert response.json()["error"]["type"] == "invalid_request_error"


def test_model_refusal_preserves_code_and_names_input(dispatch):
    dispatch.output.errors = [ExtractItemError(code=ErrorCode.INPUT_TOO_LONG.value, message="too long")]
    response = dispatch.client.post("/v1/decisions", json=FIXTURE["requests"][0]["request"])
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "INPUT_TOO_LONG"
    assert response.json()["error"]["param"] == "input"


@pytest.mark.parametrize(
    ("endpoint", "fault"),
    [
        ("decisions", "empty_output"),
        ("decisions", "oversized_predicate"),
        ("decisions", "oversized_choice"),
        ("systemone", "empty_output"),
        ("systemone", "oversized_choice"),
    ],
)
def test_malformed_worker_results_keep_the_public_error_envelope(dispatch, endpoint, fault):
    if fault == "empty_output":
        dispatch.output.entities = []
    elif fault == "oversized_predicate":
        dispatch.output.data[0]["question_0"]["noul"] = 10**4000
    else:
        dispatch.output.data[0]["question_1"]["probabilities"]["option_0"] = 10**4000
    case = FIXTURE["requests"][0]
    body = case["request" if endpoint == "decisions" else "systemone"]
    response = dispatch.client.post(f"/v1/{endpoint}", json=body)
    assert response.status_code == 500
    assert response.headers["content-type"].startswith("application/json")
    error = response.json()["error" if endpoint == "decisions" else "detail"]
    assert error["code"] == "INTERNAL_ERROR"
    assert error["message"] == "model returned an invalid decision result"
