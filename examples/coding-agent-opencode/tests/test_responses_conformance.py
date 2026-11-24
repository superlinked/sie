"""``conformance.py --api responses`` against a scripted OpenAI Responses stub.

The stub plays the wire a conforming ``/v1/responses`` sends, events and
refusals included, and each knob breaks one thing, so every verdict is checked
both for what it accepts and for what it refuses. This checks the checker;
an endpoint is checked by running ``conformance.py --api responses`` against
it.
"""

from __future__ import annotations

import json
import threading
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

import pytest

import conformance

KNOBS: dict[str, Any] = {}


def _texts(body: dict[str, Any]) -> str:
    if isinstance(body["input"], str):
        return body["input"]
    out = []
    for item in body["input"]:
        content = item.get("content")
        if isinstance(content, list):
            out += [part.get("text", "") for part in content]
    return " ".join(out)


def _refusal(body: dict[str, Any]) -> tuple[int, str, str, str] | None:
    if body.get("previous_response_id") is not None:
        param = KNOBS.get("previous_param", "previous_response_id")
        return 400, "unsupported_field", param, f"{param}: SIE does not store responses"
    for position, tool in enumerate(body.get("tools") or []):
        if tool["type"] != "function" and not KNOBS.get("accept_hosted"):
            param = f"tools[{position}].type"
            return 400, "unsupported_field", param, f"{param}: '{tool['type']}' tools are not supported"
    return None


def _reply(body: dict[str, Any]) -> dict[str, Any]:
    items = body["input"] if isinstance(body["input"], list) else []
    last = items[-1] if items else {}
    if last.get("type") == "function_call_output" and isinstance(last["output"], list):
        if KNOBS.get("image_calls_again"):
            return {"call": ("view_image", {"path": "red.png"})}
        return {"text": "Red."}
    if last.get("type") == "function_call_output":
        return {"text": "It is 18 degrees and overcast in Paris."}
    if body.get("tools") and "weather" in _texts(body):
        return {"call": ("get_weather", {"city": "Paris"})}
    return {"text": "ready"}


def _response(body: dict[str, Any], status: str, output: list[dict[str, Any]], usage: Any) -> dict[str, Any]:
    return {
        "id": "resp-stub",
        "object": "response",
        "created_at": 0,
        "model": body["model"],
        "status": status,
        "output": output,
        "usage": usage,
    }


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args: Any) -> None:
        pass

    def _json(self, status: int, payload: Any, headers: dict[str, str] | None = None) -> None:
        raw = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(raw)))
        for name, value in (headers or {}).items():
            self.send_header(name, value)
        self.end_headers()
        self.wfile.write(raw)

    def _error(self, status: int, code: str, param: str | None, message: str, **headers: str) -> None:
        error = {"message": message, "type": "invalid_request_error", "param": param, "code": code}
        self._json(status, {"error": error}, {name.replace("_", "-"): value for name, value in headers.items()})

    def do_GET(self) -> None:
        self._json(200, {"models": [{"name": "stub-model", "max_sequence_length": 131072}]})

    def do_POST(self) -> None:
        body = json.loads(self.rfile.read(int(self.headers["content-length"])))
        KNOBS["_posts"] = KNOBS.get("_posts", 0) + 1
        if KNOBS["_posts"] <= KNOBS.get("loading", 0):
            self._error(503, "MODEL_LOADING", None, "loading", retry_after="1")
            return
        if KNOBS.get("busy", 0) > len(KNOBS.setdefault("_busy", [])) and body.get("tools"):
            KNOBS["_busy"].append(body)
            self._error(503, "RESOURCE_EXHAUSTED", None, "busy", retry_after="0")
            return
        refusal = _refusal(body)
        if refusal:
            self._error(refusal[0], refusal[1], refusal[2], refusal[3])
            return
        items = body["input"] if isinstance(body["input"], list) else []
        if items and isinstance(items[-1].get("output"), list) and "image_refusal" in KNOBS:
            self._error(400, "unsupported_field", *KNOBS["image_refusal"])
            return
        reply = _reply(body)
        usage = {"input_tokens": 12, "output_tokens": 3, "total_tokens": 15}
        if "call" in reply:
            tool, arguments = reply["call"]
            item = {"type": "function_call", "id": "fc-0", "call_id": "call_0", "name": tool}
            output = [{**item, "arguments": json.dumps(arguments), "status": "completed"}]
        else:
            part = {"type": "output_text", "text": reply["text"], "annotations": []}
            output = [{"type": "message", "id": "msg-0", "role": "assistant", "status": "completed", "content": [part]}]
        if not body.get("stream"):
            self._json(200, [] if KNOBS.get("buffered_list") else _response(body, "completed", output, usage))
            return
        self.send_response(200)
        self.send_header("content-type", "text/event-stream")
        self.send_header("x-sie-served-by", "stub")
        self.end_headers()
        events: list[tuple[str, dict[str, Any]]] = [
            ("response.created", {"response": _response(body, "in_progress", [], None)}),
            ("response.in_progress", {"response": _response(body, "in_progress", [], None)}),
        ]
        done = output[0]
        if "call" in reply:
            ids = {"item_id": "fc-0", "output_index": 0}
            done_arguments = '{"city": "Lyon"}' if KNOBS.get("done_differs") else done["arguments"]
            events += [
                ("response.output_item.added", {"output_index": 0, "item": {**item, "arguments": ""}}),
                ("response.function_call_arguments.delta", {**ids, "delta": done["arguments"]}),
                ("response.function_call_arguments.done", {**ids, "arguments": done["arguments"]}),
                ("response.output_item.done", {"output_index": 0, "item": {**done, "arguments": done_arguments}}),
            ]
        else:
            ids = {"item_id": "msg-0", "output_index": 0, "content_index": 0}
            empty = {"type": "output_text", "text": "", "annotations": []}
            events += [
                ("response.output_item.added", {"output_index": 0, "item": {**done, "content": []}}),
                ("response.content_part.added", {**ids, "part": empty}),
                ("response.output_text.delta", {**ids, "delta": reply["text"]}),
                ("response.output_text.done", {**ids, "text": reply["text"]}),
                ("response.content_part.done", {**ids, "part": done["content"][0]}),
                ("response.output_item.done", {"output_index": 0, "item": done}),
            ]
        final = [{**output[0], "arguments": "{}"}] if KNOBS.get("final_differs") and "call" in reply else output
        events.append(("response.completed", {"response": _response(body, "completed", final, usage)}))
        for number, (name, data) in enumerate(events):
            if KNOBS.get("sequence_gap") and number >= 3:
                number += 1
            payload = {"type": name, "sequence_number": number, **data}
            self.wfile.write(f"event: {name}\ndata: {json.dumps(payload)}\n\n".encode())


@pytest.fixture(scope="module")
def base_url() -> Iterator[str]:
    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}/v1"
    finally:
        server.shutdown()


@pytest.fixture(autouse=True)
def reset_knobs(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    monkeypatch.setattr(conformance.time, "sleep", lambda _seconds: None)
    KNOBS.clear()
    yield
    KNOBS.clear()


def _cases(base_url: str, **knobs: Any) -> dict[str, dict[str, Any]]:
    KNOBS.update(knobs)
    report = conformance.run_responses(base_url, "", "stub-model")
    return {case["name"]: case for case in report["cases"]}


def test_a_conforming_endpoint_passes(base_url: str) -> None:
    cases = _cases(base_url)
    assert {name: case["status"] for name, case in cases.items()} == dict.fromkeys(cases, "pass")
    assert cases["function_call"]["observed"]["tool_calls"][0]["first_delta_has_id_and_name"]
    assert cases["image_in_tool_output"]["observed"]["mentions_red"] is True
    assert cases["event_order"]["observed"]["streams"] == 4


def test_an_image_refused_for_a_model_without_image_input_is_recorded(base_url: str) -> None:
    refusal = ("messages", "Model 'stub-model' does not support image input")
    case = _cases(base_url, image_refusal=refusal)["image_in_tool_output"]
    assert case["status"] == "recorded"
    assert case["detail"].startswith("the model takes no image input: 400 unsupported_field")


@pytest.mark.parametrize(
    "refusal",
    [
        ("input[3].output[0].type", "input[3].output[0].type: parts of type 'input_image' are not supported"),
        ("tools[0].parameters", "tools[0].parameters: invalid schema"),
    ],
)
def test_any_other_refusal_of_the_image_request_fails(base_url: str, refusal: tuple[str, str]) -> None:
    case = _cases(base_url, image_refusal=refusal)["image_in_tool_output"]
    assert case["status"] == "fail"
    assert refusal[1] in case["detail"]


def test_another_view_image_call_is_not_an_answer_to_the_image(base_url: str) -> None:
    case = _cases(base_url, image_calls_again=True)["image_in_tool_output"]
    assert case["status"] == "fail"
    assert "answered with another function_call: view_image" in case["detail"]
    assert "no text answer to the image" in case["detail"]


def test_an_accepted_hosted_tool_fails_the_refusal_case(base_url: str) -> None:
    case = _cases(base_url, accept_hosted=True)["hosted_tool_refusal"]
    assert case["status"] == "fail"
    assert case["detail"] == "accepted, expected a 400 at tools[1].type"


def test_a_refusal_at_another_field_fails(base_url: str) -> None:
    case = _cases(base_url, previous_param="input")["previous_response_id_refusal"]
    assert case["status"] == "fail"
    assert "400 at 'input', expected 'previous_response_id'" in case["detail"]


def test_done_arguments_that_differ_from_the_stream_fail(base_url: str) -> None:
    case = _cases(base_url, done_differs=True)["function_call"]
    assert case["status"] == "fail"
    assert "output_item.done arguments differ" in case["detail"]


def test_a_terminal_output_that_differs_from_the_stream_fails(base_url: str) -> None:
    case = _cases(base_url, final_differs=True)["function_call"]
    assert case["status"] == "fail"
    assert "the terminal response's output differs from the streamed call" in case["detail"]


def test_a_buffered_answer_that_is_not_an_object_fails_its_case(base_url: str) -> None:
    case = _cases(base_url, buffered_list=True)["buffered_response"]
    assert case["status"] == "fail"
    assert "expected a JSON object, got list" in case["detail"]


def test_a_gap_in_the_sequence_numbers_breaks_the_order(base_url: str) -> None:
    case = _cases(base_url, sequence_gap=True)["event_order"]
    assert case["status"] == "fail"
    assert "sequence_number runs" in case["detail"]


def test_busy_refusals_are_retried(base_url: str) -> None:
    case = _cases(base_url, busy=2)["function_call"]
    assert case["status"] == "pass"
    assert case["retries"] == 2


def test_the_warm_up_waits_out_a_loading_model(base_url: str) -> None:
    KNOBS.update(loading=2)
    report = conformance.run_responses(base_url, "", "stub-model")
    assert report["warm_up"]["attempts"] == 3
    assert report["summary"]["fail"] == 0


def _event(name: str, number: int, **fields: Any) -> dict[str, Any]:
    return {"event": name, "type": name, "sequence_number": number, **fields}


@pytest.mark.parametrize(
    ("events", "problem"),
    [
        ([_event("response.in_progress", 0)], "not response.created then response.in_progress"),
        (
            [
                _event("response.created", 0),
                _event("response.in_progress", 1),
                _event("response.output_text.delta", 2, output_index=0, item_id="msg-0"),
                _event("response.completed", 3),
            ],
            "response.output_text.delta for item msg-0 at output_index 0 while None is open",
        ),
        (
            [
                _event("response.created", 0),
                _event("response.in_progress", 1),
                _event("response.output_item.added", 2, output_index=0, item_id="msg-0"),
                _event("response.output_text.delta", 3, output_index=1, item_id="msg-0"),
                _event("response.output_item.done", 4, output_index=0, item_id="msg-0"),
                _event("response.completed", 5),
            ],
            "response.output_text.delta for item msg-0 at output_index 1 while msg-0 is open",
        ),
        (
            [
                _event("response.created", 0),
                _event("response.in_progress", 1),
                _event("response.output_item.added", 2, output_index=0, item_id="msg-0"),
                _event("response.completed", 3),
            ],
            "item msg-0 never done",
        ),
        (
            [_event("response.created", 0), _event("response.in_progress", 1), _event("response.output_item.added", 2)],
            "not one terminal event last",
        ),
        (
            [{**_event("response.created", 0), "type": "response.in_progress"}, _event("response.completed", 1)],
            "carries type 'response.in_progress'",
        ),
    ],
)
def test_event_order_problems(events: list[dict[str, Any]], problem: str) -> None:
    assert any(problem in found for found in conformance.responses_event_problems(events))


def test_a_failed_response_may_leave_an_item_open() -> None:
    events = [
        _event("response.created", 0),
        _event("response.in_progress", 1),
        _event("response.output_item.added", 2, output_index=0, item_id="msg-0"),
        _event("response.output_text.delta", 3, output_index=0, item_id="msg-0"),
        _event("response.failed", 4),
    ]
    assert conformance.responses_event_problems(events) == []
