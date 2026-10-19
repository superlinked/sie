"""SIEClient.system_one: System One typed decisions over extract."""

import json
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import msgpack
import pytest
from sie_sdk import SIEAsyncClient, SIEClient
from sie_sdk.client.async_ import _AioResponse
from sie_sdk.client.errors import RequestError
from sie_sdk.systemone import choice_confidence, score_confidence, system_one_request, system_one_result

FIXTURE = json.loads(
    (Path(__file__).resolve().parents[3] / "wire-fixtures" / "systemone.json").read_text(encoding="utf-8")
)
RESPONSE_CASE = FIXTURE["responses"][0]
MODEL = RESPONSE_CASE["model"]
QUESTIONS = {qid: {"type": "noul"} for qid in RESPONSE_CASE["questions"]}
# Invalid requests only the HTTP route can see: the SDK takes the model as an
# argument and sends no media fields.
ROUTE_ONLY_INVALID_REQUESTS = {"missing model", "empty model", "images", "audio", "videos"}


def _extract_response(item: dict[str, Any]) -> bytes:
    body = {"model": MODEL, "items": [item], "usage": {"input_tokens": RESPONSE_CASE["input_tokens"]}}
    return msgpack.packb(body, use_bin_type=True)


def _answers_item() -> dict[str, Any]:
    return {"id": "item-0", "entities": [], "data": RESPONSE_CASE["data"]}


def _assert_close(actual: Any, expected: Any) -> None:
    if isinstance(expected, dict):
        assert set(actual) == set(expected)
        for key, value in expected.items():
            _assert_close(actual[key], value)
    elif isinstance(expected, float):
        assert actual == pytest.approx(expected)
    else:
        assert actual == expected


def test_sync_system_one_sends_one_strict_extract_item_and_reshapes_the_answers() -> None:
    response = MagicMock(status_code=200, content=_extract_response(_answers_item()), headers={})
    with patch("sie_sdk.client.sync.httpx.Client") as http_client:
        http_client.return_value.post.return_value = response
        client = SIEClient("http://localhost:8080")
        result = client.system_one(MODEL, "I was charged twice.", QUESTIONS)
        call = http_client.return_value.post.call_args
        client.close()

    assert call.args[0] == f"/v1/extract/{MODEL}"
    body = msgpack.unpackb(call.kwargs["content"], raw=False)
    assert body["items"] == [{"text": "I was charged twice."}]
    assert body["params"]["options"]["overflow_policy"] == "error"
    assert body["params"]["output_schema"] == {qid: {"type": "noul", "instructions": ""} for qid in QUESTIONS}
    expected = dict(RESPONSE_CASE["response"])
    # Usage is the metered count the response reports in its X-SIE-Units-*
    # headers; this response carries none.
    expected["usage"] = {"input_tokens": 0, "output_tokens": 0}
    _assert_close(result, expected)


def test_system_one_reports_metered_input_tokens() -> None:
    response = MagicMock(
        status_code=200,
        content=_extract_response(_answers_item()),
        headers={"X-SIE-Units-Input-Tokens": "42"},
    )
    with patch("sie_sdk.client.sync.httpx.Client") as http_client:
        http_client.return_value.post.return_value = response
        client = SIEClient("http://localhost:8080")
        result = client.system_one(MODEL, "I was charged twice.", QUESTIONS)
        client.close()
    assert result["usage"] == {"input_tokens": 42, "output_tokens": 0}


@pytest.mark.asyncio
async def test_async_system_one_matches_sync() -> None:
    client = SIEAsyncClient("http://localhost:8080")
    post = AsyncMock(return_value=_AioResponse(200, _extract_response(_answers_item()), {}))
    with patch.object(client, "_post", post):
        result = await client.system_one(MODEL, {"subject": "refund"}, QUESTIONS)
    body = msgpack.unpackb(post.call_args.kwargs["data"], raw=False)
    assert body["items"] == [{"metadata": {"state": {"subject": "refund"}}}]
    assert result["answers"]["intent"]["confidence"] == pytest.approx(0.7)
    assert result["answers"]["urgency"]["confidence"] == pytest.approx(0.55)


def test_state_that_does_not_fit_raises() -> None:
    item = {"id": "item-0", "entities": [], "data": {}, "error": {"code": "INPUT_TOO_LONG", "message": "does not fit"}}
    response = MagicMock(status_code=200, content=_extract_response(item), headers={})
    with patch("sie_sdk.client.sync.httpx.Client") as http_client:
        http_client.return_value.post.return_value = response
        client = SIEClient("http://localhost:8080")
        with pytest.raises(RequestError, match="does not fit") as raised:
            client.system_one(MODEL, "x" * 10, QUESTIONS)
        client.close()
    assert raised.value.code == "INPUT_TOO_LONG"


@pytest.mark.parametrize("case", FIXTURE["requests"], ids=lambda case: case["name"])
def test_fixture_requests(case: dict[str, Any]) -> None:
    item, schema = system_one_request(case["request"]["state"], case["request"]["questions"])
    assert item == case["extract"]["item"]
    assert schema == case["extract"]["output_schema"]


@pytest.mark.parametrize(
    "case",
    [case for case in FIXTURE["invalid_requests"] if case["name"] not in ROUTE_ONLY_INVALID_REQUESTS],
    ids=lambda case: case["name"],
)
def test_fixture_invalid_requests(case: dict[str, Any]) -> None:
    request = case["request"]
    with pytest.raises(RequestError):
        system_one_request(request.get("state"), request.get("questions", {}))


@pytest.mark.parametrize("kind", ["choice", "score"])
def test_fixture_confidence(kind: str) -> None:
    fn = choice_confidence if kind == "choice" else score_confidence
    for case in FIXTURE["confidence"][kind]:
        assert fn(case["probabilities"]) == pytest.approx(case["confidence"])


@pytest.mark.parametrize("case", FIXTURE["responses"], ids=lambda case: case["name"])
def test_fixture_responses(case: dict[str, Any]) -> None:
    questions = {qid: {"type": "noul"} for qid in case["questions"]}
    extract = {
        "id": "item-0",
        "entities": [],
        "data": case["data"],
        "request": {"usage": {"input_tokens": case["input_tokens"]}},
    }
    _assert_close(system_one_result(case["model"], extract, questions), case["response"])  # type: ignore[arg-type]


def test_missing_answer_raises_invalid_response() -> None:
    extract = {"id": "item-0", "entities": [], "data": {}}
    with pytest.raises(RequestError) as raised:
        system_one_result(MODEL, extract, {"q": {"type": "noul"}})  # type: ignore[arg-type]
    assert raised.value.code == "INVALID_RESPONSE"


@pytest.mark.parametrize("probabilities", [{"0": 0.5, "high": 0.5}, {"0": 0.5, "1": "half"}])
def test_unreadable_probabilities_raise_invalid_response(probabilities: dict[str, Any]) -> None:
    extract = {"id": "item-0", "entities": [], "data": {"q": {"type": "score", "probabilities": probabilities}}}
    with pytest.raises(RequestError) as raised:
        system_one_result(MODEL, extract, {"q": {"type": "score"}})  # type: ignore[arg-type]
    assert raised.value.code == "INVALID_RESPONSE"
