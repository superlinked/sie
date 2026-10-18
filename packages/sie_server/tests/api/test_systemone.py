"""Tests for the System One-compatible POST /v1/systemone."""

import asyncio
import json
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient
from sie_server.adapters.base import ModelCapabilities, ModelDims
from sie_server.adapters.errors import InputTooLongError
from sie_server.adapters.pplx_decider.questions import answer as pplx_answer
from sie_server.adapters.pplx_decider.questions import parse_questions as pplx_questions
from sie_server.api.systemone import (
    MAX_RETRY_AFTER_S,
    choice_confidence,
    error_response,
    normalize_questions,
    score_confidence,
)
from sie_server.api.systemone import router as systemone_router
from sie_server.config.model import EncodeTask, ExtractTask, ModelConfig, ProfileConfig, Tasks
from sie_server.core.inference_output import ExtractItemError, ExtractOutput
from sie_server.core.registry import ModelRegistry
from sie_server.core.timing import RequestTiming
from sie_server.core.worker import QueueFullError, WorkerResult
from sie_server.types.responses import ErrorCode

DECIDE = "fastino/GLiNER2.5-Decide"
EMBEDDER = "org/embedder"
DECIDE_ADAPTER = "sie_server.adapters.gliner2.decide:GLiNER2DecideAdapter"
PPLX_ADAPTER = "sie_server.adapters.pplx_decider.adapter:PerplexityDecisionAdapter"

QUESTIONS: dict[str, Any] = {
    "spam": {"type": "noul", "instructions": "Is this message spam?"},
    "intent": {
        "type": "choice",
        "instructions": "What does the sender want?",
        "criteria": {"billing": "payments and refunds", "support": None, "other": None},
    },
    "urgency": {
        "type": "score",
        "instructions": "How urgent is it?",
        "criteria": ["Can wait", "Needs attention this week", "Needs attention today"],
    },
}


def _answer(question: dict[str, Any]) -> dict[str, Any]:
    """An adapter answer in the shape the decision adapters return."""
    if question["type"] == "noul":
        return {"type": "noul", "noul": 0.9, "answer": True, "confidence": 0.9}
    if question["type"] == "choice":
        keys = list(question["criteria"])
        probabilities = {key: (0.8 if i == 0 else 0.2 / (len(keys) - 1)) for i, key in enumerate(keys)}
        # The adapters' own (entropy) confidence, which the route replaces.
        return {"type": "choice", "choice": keys[0], "probabilities": probabilities, "confidence": 0.123}
    levels = question["criteria"]
    probabilities = {str(i): (0.8 if i == len(levels) - 1 else 0.2 / (len(levels) - 1)) for i in range(len(levels))}
    score = sum(i * p for i, p in enumerate(probabilities.values()))
    return {
        "type": "score",
        "score": score,
        "legend": {str(i): level for i, level in enumerate(levels)},
        "probabilities": probabilities,
        "confidence": 0.123,
    }


def _decide(items: list[Any], *, output_schema: dict[str, Any] | None = None, **_: Any) -> ExtractOutput:
    data = [{qid: _answer(q) for qid, q in (output_schema or {}).items()} for _ in items]
    return ExtractOutput(entities=[[] for _ in items], data=data, input_token_counts=[42 for _ in items])


@pytest.fixture
def adapter() -> MagicMock:
    adapter = MagicMock()
    adapter.extract = MagicMock(side_effect=_decide)
    adapter.capabilities = ModelCapabilities(inputs=["text"], outputs=[])
    adapter.dims = ModelDims()
    return adapter


def _worker(adapter: MagicMock) -> MagicMock:
    worker = MagicMock()

    async def submit_extract(
        prepared_items, items, *, labels=None, output_schema=None, instruction=None, options=None, timing=None
    ):
        future = asyncio.get_running_loop().create_future()
        try:
            output = adapter.extract(
                items, labels=labels, output_schema=output_schema, instruction=instruction, options=options
            )
        except Exception as exc:  # noqa: BLE001 — the worker hands adapter errors back through the future
            future.set_exception(exc)
            return future
        request_timing = timing or RequestTiming()
        if request_timing._inference_start is None:
            request_timing._inference_start = request_timing._queue_start or 0
            request_timing._inference_end = request_timing._inference_start
        future.set_result(WorkerResult(output=output, timing=request_timing))
        return future

    worker.submit_extract = submit_extract
    return worker


@pytest.fixture(params=[DECIDE_ADAPTER, PPLX_ADAPTER])
def registry(adapter: MagicMock, request: pytest.FixtureRequest) -> MagicMock:
    configs = {
        DECIDE: ModelConfig(
            sie_id=DECIDE,
            hf_id=DECIDE,
            tasks=Tasks(extract=ExtractTask()),
            profiles={"default": ProfileConfig(adapter_path=request.param, max_batch_tokens=8192)},
        ),
        EMBEDDER: ModelConfig(
            sie_id=EMBEDDER,
            hf_id=EMBEDDER,
            tasks=Tasks(encode=EncodeTask()),
            profiles={"default": ProfileConfig(adapter_path="test:Embedder", max_batch_tokens=8192)},
        ),
    }
    registry = MagicMock(spec=ModelRegistry)
    registry.has_model.side_effect = lambda name: name in configs
    registry.get_config.side_effect = lambda name: configs[name]
    registry.is_loaded.return_value = True
    registry.is_loading.return_value = False
    registry.is_unloading.return_value = False
    registry.is_failed.return_value = False
    registry.get_failure.return_value = None
    registry.get.return_value = adapter
    registry.model_names = list(configs)
    registry.device = "cpu"
    registry.start_worker = AsyncMock(return_value=_worker(adapter))
    return registry


@pytest.fixture
def client(registry: MagicMock) -> TestClient:
    app = FastAPI()
    app.include_router(systemone_router)
    app.state.registry = registry
    return TestClient(app)


def _post(client: TestClient, **body: Any) -> Any:
    payload = {"model": DECIDE, "state": "I was charged twice. Please fix this ASAP.", "questions": QUESTIONS, **body}
    return client.post("/v1/systemone", json=payload)


class TestAnswers:
    @pytest.mark.parametrize("probability", [0.0, 0.2, 0.5, 0.8, 1.0])
    def test_native_perplexity_noul_includes_system_one_fields(self, client, adapter, probability):
        def extract(items, *, output_schema, **kwargs):
            question = pplx_questions(output_schema)[0]
            return ExtractOutput(
                entities=[[] for _ in items],
                data=[{question.id: pplx_answer(question, [1 - probability, probability])} for _ in items],
                input_token_counts=[42 for _ in items],
            )

        adapter.extract.side_effect = extract
        response = _post(client, questions={"spam": QUESTIONS["spam"]})
        assert response.status_code == 200
        assert response.json()["answers"]["spam"] == {
            "type": "noul",
            "noul": probability,
            "answer": probability > 0.5,
            "confidence": max(probability, 1 - probability),
        }

    def test_envelope(self, client: TestClient) -> None:
        response = _post(client)
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("application/json")
        body = response.json()
        assert set(body) == {"model", "answers", "usage"}
        assert body["model"] == DECIDE
        assert list(body["answers"]) == ["spam", "intent", "urgency"]
        assert body["usage"] == {"input_tokens": 42, "output_tokens": 0}
        assert all(type(v) is int for v in body["usage"].values())

    def test_json_even_when_msgpack_is_accepted(self, client: TestClient) -> None:
        payload = {"model": DECIDE, "state": "hi", "questions": QUESTIONS}
        response = client.post("/v1/systemone", json=payload, headers={"Accept": "application/msgpack"})
        assert response.headers["content-type"].startswith("application/json")

    def test_answer_shapes(self, client: TestClient) -> None:
        answers = _post(client).json()["answers"]
        assert answers["spam"]["type"] == "noul"
        assert answers["spam"]["noul"] == pytest.approx(0.9)
        assert answers["intent"]["choice"] == "billing"
        assert set(answers["intent"]["probabilities"]) == {"billing", "support", "other"}
        urgency = answers["urgency"]
        assert set(urgency["legend"]) == set(urgency["probabilities"]) == {"0", "1", "2"}
        assert urgency["legend"]["2"] == "Needs attention today"
        assert all(value is not None for value in urgency["legend"].values())

    def test_confidences_follow_system_one(self, client: TestClient) -> None:
        answers = _post(client).json()["answers"]
        # p_max 0.8 over 3 options: (0.8 - 1/3) / (1 - 1/3) = 0.7
        assert answers["intent"]["confidence"] == pytest.approx(0.7)
        # probabilities (0.1, 0.1, 0.8): spread 0.3, even 2/3, 1 - 0.45 = 0.55
        assert answers["urgency"]["confidence"] == pytest.approx(0.55)

    @pytest.mark.parametrize(
        ("state", "text", "metadata"),
        [
            ("plain text", "plain text", None),
            (
                {"subject": "refund", "body": "charged twice"},
                None,
                {"state": {"subject": "refund", "body": "charged twice"}},
            ),
            (["hi", "I was charged twice"], None, {"state": ["hi", "I was charged twice"]}),
        ],
    )
    def test_state_shapes(self, client: TestClient, adapter: MagicMock, state: Any, text: Any, metadata: Any) -> None:
        assert _post(client, state=state).status_code == 200
        item = adapter.extract.call_args.args[0][0]
        assert item.text == text
        assert item.metadata == metadata

    def test_strict_overflow_and_questions_reach_the_adapter(self, client: TestClient, adapter: MagicMock) -> None:
        _post(client)
        kwargs = adapter.extract.call_args.kwargs
        assert kwargs["options"]["overflow_policy"] == "error"
        assert list(kwargs["output_schema"]) == ["spam", "intent", "urgency"]

    @pytest.mark.parametrize("question", [{"type": "noul"}, {"type": "noul", "instructions": None}])
    def test_instructions_optional(self, client: TestClient, adapter: MagicMock, question: dict[str, Any]) -> None:
        assert _post(client, questions={"spam": question}).status_code == 200
        assert adapter.extract.call_args.kwargs["output_schema"]["spam"]["instructions"] == ""

    def test_unknown_fields_ignored(self, client: TestClient) -> None:
        questions = {"spam": {"type": "noul", "instructions": "spam?", "weight": 2}}
        response = _post(client, questions=questions, metadata={"user": "u1"}, temperature=0)
        assert response.status_code == 200


class TestValidation:
    def _assert_422(self, response: Any) -> list[dict[str, Any]]:
        assert response.status_code == 422
        detail = response.json()["detail"]
        assert isinstance(detail, list)
        assert detail
        assert all("msg" in entry and "loc" in entry for entry in detail)
        return detail

    @pytest.mark.parametrize(
        "body",
        [
            {"state": "hi", "questions": QUESTIONS},
            {"model": DECIDE, "questions": QUESTIONS},
            {"model": DECIDE, "state": "hi"},
            {"model": DECIDE, "state": "hi", "questions": {}},
            {"model": DECIDE, "state": 3, "questions": QUESTIONS},
            {"model": DECIDE, "state": None, "questions": QUESTIONS},
        ],
    )
    def test_malformed_request(self, client: TestClient, body: dict[str, Any]) -> None:
        self._assert_422(client.post("/v1/systemone", json=body))

    def test_malformed_json(self, client: TestClient) -> None:
        response = client.post("/v1/systemone", content=b"{", headers={"Content-Type": "application/json"})
        self._assert_422(response)

    @pytest.mark.parametrize("field", ["images", "audio", "videos"])
    def test_media_rejected(self, client: TestClient, field: str) -> None:
        self._assert_422(_post(client, **{field: ["https://example.com/x.png"]}))

    def test_score_level_limit(self, client: TestClient) -> None:
        questions = {"q": {"type": "score", "instructions": "?", "criteria": [str(i) for i in range(11)]}}
        detail = self._assert_422(_post(client, questions=questions))
        assert "at most 10" in detail[0]["msg"]

    def test_score_null_level(self, client: TestClient) -> None:
        questions = {"q": {"type": "score", "instructions": "?", "criteria": ["low", None]}}
        self._assert_422(_post(client, questions=questions))

    def test_question_not_an_object(self, client: TestClient) -> None:
        self._assert_422(_post(client, questions={"q": "spam?"}))

    def test_model_without_typed_questions(self, client: TestClient) -> None:
        detail = self._assert_422(_post(client, model=EMBEDDER))
        assert "does not answer typed questions" in detail[0]["msg"]

    def test_adapter_rejects_question(self, client: TestClient, adapter: MagicMock) -> None:
        adapter.extract.side_effect = ValueError("question 'q': unknown type 'maybe'")
        detail = self._assert_422(_post(client))
        assert "unknown type" in detail[0]["msg"]

    def test_questions_over_token_budget(self, client: TestClient, adapter: MagicMock) -> None:
        adapter.extract.side_effect = InputTooLongError("questions and labels take 300 tokens; at most 256")
        detail = self._assert_422(_post(client))
        assert "300 tokens" in detail[0]["msg"]

    def test_state_does_not_fit(self, client: TestClient, adapter: MagicMock) -> None:
        def too_long(items: list[Any], **_: Any) -> ExtractOutput:
            error = ExtractItemError(code=ErrorCode.INPUT_TOO_LONG.value, message="The document does not fit")
            return ExtractOutput(entities=[[]], data=[{}], errors=[error], input_token_counts=[0])

        adapter.extract.side_effect = too_long
        detail = self._assert_422(_post(client))
        assert detail[0]["loc"] == ["body", "state"]
        assert detail[0]["code"] == ErrorCode.INPUT_TOO_LONG.value

    def test_unreadable_state_keeps_its_code(self, client: TestClient, adapter: MagicMock) -> None:
        def unreadable(items: list[Any], **_: Any) -> ExtractOutput:
            error = ExtractItemError(code=ErrorCode.INVALID_INPUT.value, message="state is not valid JSON")
            return ExtractOutput(entities=[[]], data=[{}], errors=[error], input_token_counts=[0])

        adapter.extract.side_effect = unreadable
        detail = self._assert_422(_post(client))
        assert detail[0]["code"] == ErrorCode.INVALID_INPUT.value


class TestErrors:
    def test_unknown_model(self, client: TestClient) -> None:
        response = _post(client, model="org/missing")
        assert response.status_code == 404
        detail = response.json()["detail"]
        assert detail["error_type"] == "not_found_error"
        assert "org/missing" in detail["message"]

    def test_missing_answer_is_a_server_error(self, client: TestClient, adapter: MagicMock) -> None:
        adapter.extract.side_effect = lambda items, **_: ExtractOutput(entities=[[]], data=[{}], input_token_counts=[1])
        response = _post(client)
        assert response.status_code == 500
        assert response.json()["detail"]["error_type"] == "api_error"

    @pytest.mark.parametrize("probabilities", [{"0": 0.5, "high": 0.5}, {"0": 0.5, "1": "half"}])
    def test_unreadable_answer_is_a_server_error(
        self, client: TestClient, adapter: MagicMock, probabilities: dict[str, Any]
    ) -> None:
        answer = {"type": "score", "score": 0.5, "probabilities": probabilities}
        adapter.extract.side_effect = lambda items, **_: ExtractOutput(
            entities=[[]], data=[{"q": answer}], input_token_counts=[1]
        )
        response = _post(client, questions={"q": {"type": "score", "criteria": ["low", "high"]}})
        assert response.status_code == 500
        assert response.json()["detail"]["error_type"] == "api_error"

    def test_queue_full_is_retryable(self, client: TestClient, adapter: MagicMock) -> None:
        adapter.extract.side_effect = QueueFullError("queue full")
        response = _post(client)
        assert response.status_code == 503
        assert int(response.headers["Retry-After"]) <= MAX_RETRY_AFTER_S
        assert response.headers["retry-after-ms"] == str(int(float(response.headers["Retry-After"]) * 1000))
        assert set(response.json()["detail"]) >= {"error_type", "message"}

    def test_retry_after_capped_and_mirrored(self) -> None:
        error = HTTPException(
            status_code=503,
            detail={"code": "MODEL_LOADING", "message": "loading"},
            headers={"Retry-After": "270"},
        )
        response = error_response(error)
        assert response.status_code == 503
        assert response.headers["Retry-After"] == str(MAX_RETRY_AFTER_S)
        assert response.headers["retry-after-ms"] == str(MAX_RETRY_AFTER_S * 1000)

    def test_short_retry_after_kept(self) -> None:
        error = HTTPException(
            status_code=503, detail={"code": "QUEUE_FULL", "message": "busy"}, headers={"Retry-After": "5"}
        )
        response = error_response(error)
        assert response.headers["Retry-After"] == "5"
        assert response.headers["retry-after-ms"] == "5000"

    def test_bad_request_becomes_422(self) -> None:
        response = error_response(HTTPException(status_code=400, detail={"code": "INVALID_INPUT", "message": "bad"}))
        assert response.status_code == 422
        assert json.loads(response.body)["detail"][0]["code"] == "INVALID_INPUT"

    def test_bad_request_without_a_code_has_no_code(self) -> None:
        response = error_response(HTTPException(status_code=400, detail="bad"))
        assert "code" not in json.loads(response.body)["detail"][0]

    def test_unparseable_retry_after_passes_through(self) -> None:
        date = "Wed, 21 Oct 2026 07:28:00 GMT"
        error = HTTPException(
            status_code=503, detail={"code": "QUEUE_FULL", "message": "busy"}, headers={"Retry-After": date}
        )
        response = error_response(error)
        assert response.headers["Retry-After"] == date
        assert "retry-after-ms" not in response.headers


class TestConfidence:
    @pytest.mark.parametrize(
        ("probabilities", "expected"),
        [([1.0, 0.0], 1.0), ([0.5, 0.5], 0.0), ([0.8, 0.1, 0.1], 0.7), ([1.0], 1.0)],
    )
    def test_choice(self, probabilities: list[float], expected: float) -> None:
        assert choice_confidence(probabilities) == pytest.approx(expected)

    @pytest.mark.parametrize(
        ("probabilities", "expected"),
        [([0.0, 0.0, 1.0], 1.0), ([0.1, 0.1, 0.8], 0.55), ([0.5, 0.0, 0.5], 0.0), ([1.0], 1.0)],
    )
    def test_score(self, probabilities: list[float], expected: float) -> None:
        assert score_confidence(probabilities) == pytest.approx(expected)


def test_normalize_questions_keeps_other_fields() -> None:
    questions = {"q": {"type": "noul", "criteria": {"true": "yes"}, "labels": {"false": "ham", "true": "spam"}}}
    assert normalize_questions(questions)["q"] == {
        "type": "noul",
        "criteria": {"true": "yes"},
        "labels": {"false": "ham", "true": "spam"},
        "instructions": "",
    }


# ---------------------------------------------------------------------------- shared wire fixture

FIXTURE = json.loads(
    (Path(__file__).resolve().parents[3] / "wire-fixtures" / "systemone.json").read_text(encoding="utf-8")
)


@pytest.mark.parametrize("case", FIXTURE["requests"], ids=lambda case: case["name"])
def test_fixture_requests(client: TestClient, adapter: MagicMock, case: dict[str, Any]) -> None:
    assert client.post("/v1/systemone", json=case["request"]).status_code == 200
    expected = case["extract"]
    items = adapter.extract.call_args.args[0]
    kwargs = adapter.extract.call_args.kwargs
    assert len(items) == 1
    assert items[0].text == expected["item"].get("text")
    assert items[0].metadata == expected["item"].get("metadata")
    assert kwargs["output_schema"] == expected["output_schema"]
    assert kwargs["options"]["overflow_policy"] == expected["options"]["overflow_policy"]


@pytest.mark.parametrize("case", FIXTURE["invalid_requests"], ids=lambda case: case["name"])
def test_fixture_invalid_requests(client: TestClient, adapter: MagicMock, case: dict[str, Any]) -> None:
    response = client.post("/v1/systemone", json=case["request"])
    assert response.status_code == 422
    assert isinstance(response.json()["detail"], list)
    adapter.extract.assert_not_called()


@pytest.mark.parametrize("case", FIXTURE["responses"], ids=lambda case: case["name"])
def test_fixture_responses(client: TestClient, adapter: MagicMock, case: dict[str, Any]) -> None:
    adapter.extract.side_effect = lambda items, **_: ExtractOutput(
        entities=[[]], data=[case["data"]], input_token_counts=[case["input_tokens"]]
    )
    questions = {qid: {"type": "noul"} for qid in case["questions"]}
    body = client.post("/v1/systemone", json={"model": case["model"], "state": "x", "questions": questions}).json()
    assert _approx_equal(body, case["response"])


def _approx_equal(actual: Any, expected: Any) -> bool:
    if isinstance(expected, dict):
        assert isinstance(actual, dict)
        assert set(actual) == set(expected)
        return all(_approx_equal(actual[key], value) for key, value in expected.items())
    if isinstance(expected, float):
        assert actual == pytest.approx(expected)
        return True
    assert actual == expected
    assert type(actual) is type(expected)
    return True


@pytest.mark.parametrize("kind", ["choice", "score"])
def test_fixture_confidence(kind: str) -> None:
    fn = choice_confidence if kind == "choice" else score_confidence
    for case in FIXTURE["confidence"][kind]:
        assert fn(case["probabilities"]) == pytest.approx(case["confidence"])


@pytest.mark.parametrize("case", FIXTURE["retry_after"], ids=lambda case: case["retry_after"])
def test_fixture_retry_after(case: dict[str, str]) -> None:
    error = HTTPException(
        status_code=503, detail={"code": "MODEL_LOADING", "message": "x"}, headers={"Retry-After": case["retry_after"]}
    )
    response = error_response(error)
    assert response.headers["Retry-After"] == case["expected"]
    assert response.headers["retry-after-ms"] == case["expected_ms"]
