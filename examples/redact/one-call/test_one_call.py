"""Weightless checks of complete-input, masking and actual SDK wire behavior."""

import hashlib
import importlib.util
import json
from pathlib import Path
from typing import Any

import httpx
import msgpack
import pytest

DIRECTORY = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("redact_one_call", DIRECTORY / "run.py")
assert SPEC is not None and SPEC.loader is not None
example = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(example)
BASE = "https://sie.example.test/prefix"


def entity(text: str, start: int, end: int, label: str = "person") -> dict[str, Any]:
    return {"start": start, "end": end, "text": text[start:end], "label": label, "score": 0.9}


def response(items: Any) -> httpx.Response:
    return httpx.Response(
        200,
        content=msgpack.packb({"model": example.MODEL, "items": items}, use_bin_type=True),
        headers={"Content-Type": "application/msgpack"},
    )


def test_complete_input_and_pins() -> None:
    data = (DIRECTORY / "input.txt").read_bytes()
    text = example.load_input()
    assert hashlib.sha256(data).hexdigest() == "94ec9f7160c81390866958d9b33c2c5a9e98f854782eafc938f4e57418adfc62"
    assert (len(data), len(text)) == (963, 962)
    assert "Alejandra Santamaría-Castilla" in text
    fixture = DIRECTORY.parents[2] / "packages/sie_server/tests/adapters/fixtures/gliner_union/redact-documents.json"
    documents = json.loads(fixture.read_bytes())["documents"]
    original = next(row for row in documents if row["index"] == 240)
    assert data == original["text"].encode("utf-8")
    source = json.loads((DIRECTORY / "source.json").read_text(encoding="utf-8"))
    assert source["input"]["uid"] == "gretel-en-34966"
    assert source["logical_request"]["body"] == example.logical_request(text)["body"]
    assert "labels" not in source["logical_request"]["body"]["params"]
    assert len(source["expected_model_configuration"]["runtime"]["default_labels"]) == 36


@pytest.mark.parametrize("mode", ["--show", "--saved-composition"])
def test_offline_modes_do_not_dispatch(mode: str, monkeypatch: pytest.MonkeyPatch, capsys: Any) -> None:
    def fail_if_called(*args: Any, **kwargs: Any) -> None:
        pytest.fail("offline mode attempted extraction")

    monkeypatch.setattr(example, "live_extract", fail_if_called)
    assert example.main([mode]) == 0
    output = json.loads(capsys.readouterr().out)
    if mode == "--show":
        assert output["evidence"] == "logical request display; no HTTP request sent"
        assert output["body"]["items"] == [{"text": example.load_input()}]
    else:
        assert output["evidence"] == example.SAVED_EVIDENCE
        assert len(output["entities"]) == 15
        assert len(output["merged_intervals"]) == 13
        assert len(output["masked_text"]) == len(example.load_input())


def test_unicode_overlap_union_preserves_original_code_points() -> None:
    text = "A😀e\u0301 Santamaría Z!"
    spans = [entity(text, 5, 10), entity(text, 1, 3), entity(text, 2, 4), entity(text, 1, 3)]
    spans += [entity(text, 6, 8), entity(text, 9, 12), entity(text, 12, 14)]
    output = example.render_result(text, {"entities": spans}, "test")
    assert output["merged_intervals"] == [[1, 4], [5, 14]]
    assert output["masked_text"] == "A███ █████████a Z!"
    assert len(output["masked_text"]) == len(text)
    assert len(text.encode("utf-8")) != len(text)
    assert len(text.encode("utf-16-le")) // 2 != len(text)
    # The combining mark is its own code point, even beside the masked base letter.
    assert (
        example.render_result(text, {"entities": [entity(text, 2, 3)]}, "test")["masked_text"]
        == "A😀█\u0301 Santamaría Z!"
    )


@pytest.mark.parametrize(
    "bad",
    [
        {"start": True, "end": 2},
        {"start": 0, "end": False},
        {"start": -1, "end": 2},
        {"start": 0, "end": 4},
        {"start": 2, "end": 2},
        {"start": 2, "end": 1},
        {"start": 0.0, "end": 1},
        {"start": None, "end": 1},
        {"start": 0, "end": 1, "text": "z"},
        {"start": 0, "end": 1, "label": None},
    ],
)
def test_invalid_span_fails(bad: dict[str, Any]) -> None:
    with pytest.raises(example.ExampleError):
        example.render_result("abc", {"entities": [{**entity("abc", 0, 1), **bad}]}, "test")


@pytest.mark.parametrize(
    "item",
    [{}, {"entities": None}, {"entities": {}}, {"entities": [None]}, {"entities": [], "error": {"code": "FAILED"}}],
)
def test_incomplete_or_failed_item_is_not_clean(item: Any) -> None:
    with pytest.raises(example.ExampleError):
        example.render_result("abc", item, "test")


def test_genuine_empty_entities_are_explicit() -> None:
    output = example.render_result("abc", {"entities": []}, "test")
    assert output["span_status"] == "no spans returned"
    assert output["masked_text"] == "abc"
    assert output["merged_intervals"] == []


@pytest.mark.parametrize("single_attempt", [False, True])
@pytest.mark.parametrize("no_auth", [False, True])
def test_actual_sdk_one_post_msgpack_default_profile_and_auth(
    single_attempt: bool, no_auth: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    text = example.load_input()
    calls = []
    monkeypatch.setenv("SIE_API_KEY", "test-only-key")

    def handle(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        return response([{"entities": [entity(text, 378, 407)]}])

    result = example.live_extract(
        text, BASE, single_attempt=single_attempt, no_auth=no_auth, transport=httpx.MockTransport(handle)
    )
    assert len(calls) == 1
    assert calls[0].method == "POST"
    assert str(calls[0].url) == BASE + example.EXTRACT_PATH
    assert calls[0].headers["Content-Type"] == "application/msgpack"
    assert msgpack.unpackb(calls[0].content, raw=False) == example.logical_request(text)["body"]
    assert ("Authorization" in calls[0].headers) is not no_auth
    if not no_auth:
        assert calls[0].headers["Authorization"] == "Bearer test-only-key"
    output = example.render_result(text, result, "live SDK response")
    assert output["entities"][0]["text"] == "Alejandra Santamaría-Castilla"
    assert "test-only-key" not in json.dumps(output)


@pytest.mark.parametrize("code", ["MODEL_LOADING", "QUEUE_FULL", "RESOURCE_EXHAUSTED"])
def test_single_attempt_known_failure_is_retained_without_retry(code: str) -> None:
    calls = []

    def handle(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        return httpx.Response(
            503, json={"error": {"code": code, "message": "retry later"}}, headers={"Retry-After": "0"}
        )

    with pytest.raises(example.SingleAttemptError, match="HTTP 503; no resend"):
        example.live_extract("abc", BASE, single_attempt=True, transport=httpx.MockTransport(handle))
    assert len(calls) == 1


def test_transport_uncertainty_has_no_resend_or_masked_output(monkeypatch: pytest.MonkeyPatch, capsys: Any) -> None:
    calls = []

    def handle(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        raise httpx.ReadError("test-only-secret")

    original = example.live_extract

    def extract(text: str, url: str, **kwargs: Any) -> Any:
        return original(text, url, transport=httpx.MockTransport(handle), **kwargs)

    monkeypatch.setattr(example, "live_extract", extract)
    assert example.main(["--url", BASE, "--single-attempt"]) == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "No masked output produced" in captured.err
    assert "test-only-secret" not in captured.err
    assert len(calls) == 1


@pytest.mark.parametrize(
    "item", [{}, {"entities": {}}, {"entities": [], "error": {"code": "FAILED", "message": "failed"}}]
)
def test_missing_entities_or_item_error_on_actual_wire_fails(item: Any) -> None:
    with pytest.raises(example.ExampleError):
        example.live_extract("abc", BASE, transport=httpx.MockTransport(lambda request: response([item])))


def test_ordinary_mode_keeps_cold_model_handling(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = []

    def handle(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        if len(calls) == 1:
            return httpx.Response(
                503, json={"error": {"code": "MODEL_LOADING", "message": "loading"}}, headers={"Retry-After": "0"}
            )
        return response([{"entities": []}])

    monkeypatch.setattr("sie_sdk.client.sync.time.sleep", lambda delay: None)
    assert example.live_extract("abc", BASE, transport=httpx.MockTransport(handle))["entities"] == []
    assert len(calls) == 2


def test_single_attempt_blocks_second_post_before_transport() -> None:
    calls = []
    guard = example.SingleAttempt(BASE)
    with httpx.Client(
        transport=httpx.MockTransport(lambda request: calls.append(request) or response([{"entities": []}])),
        event_hooks={"request": [guard.on_request]},
    ) as client:
        client.post(BASE + example.EXTRACT_PATH)
        with pytest.raises(example.SingleAttemptError, match="second POST"):
            client.post(BASE + example.EXTRACT_PATH)
    assert len(calls) == 1


def test_single_attempt_normalizes_url_and_allows_sdk_completion_get() -> None:
    calls = []

    def handle(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        if request.method == "POST":
            return httpx.Response(303, headers={"Location": "/result?__modal_attempt_token=test-only-token"})
        return response([{"entities": []}])

    result = example.live_extract(
        "abc",
        "HTTPS://SIE.EXAMPLE.TEST:443/prefix/",
        single_attempt=True,
        transport=httpx.MockTransport(handle),
    )
    assert result["entities"] == []
    assert [request.method for request in calls] == ["POST", "GET"]
    assert str(calls[0].url) == BASE + example.EXTRACT_PATH
    assert all(request.url.host == "sie.example.test" for request in calls)


@pytest.mark.parametrize(
    "url", ["localhost:8080", "http://user:password@example.test", "https://example.test?key=secret"]
)
def test_invalid_or_credential_bearing_url_is_rejected(url: str) -> None:
    with pytest.raises(example.ExampleError):
        example.validate_url(url)
