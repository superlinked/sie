"""``conformance.py --api messages`` against a scripted Anthropic Messages stub.

The stub plays the wire a conforming ``/v1/messages`` sends, events and
refusals included, and each knob breaks one thing, so every verdict is checked
both for what it accepts and for what it refuses. This checks the checker;
an endpoint is checked by running ``conformance.py --api messages`` against it.
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

EFFORT = "output_config.effort: Extra inputs are not permitted"
SYSTEM_TURN = 'messages.1.role: role "system" is not supported in messages; use the top-level system parameter'


def _words(text: str) -> int:
    return len(text.split())


def _texts(body: dict[str, Any]) -> str:
    out = []
    for message in body["messages"]:
        content = message["content"]
        if isinstance(content, str):
            out.append(content)
            continue
        for block in content:
            if block.get("type") == "text":
                out.append(block["text"])
    return " ".join(out)


def _refusal(body: dict[str, Any]) -> tuple[int, str, str] | None:
    if body["model"] == conformance.MISSING_MODEL:
        return 404, "not_found_error", f"Model '{body['model']}' not found"
    if "output_config" in body and not KNOBS.get("effort_ok"):
        return 400, "invalid_request_error", EFFORT
    if any(message["role"] == "system" for message in body["messages"]):
        return 400, "invalid_request_error", KNOBS.get("system_wording", SYSTEM_TURN)
    return None


def _reply(body: dict[str, Any]) -> dict[str, Any]:
    last = body["messages"][-1]["content"]
    text = _texts(body)
    thinking = (body.get("thinking") or {}).get("type") in ("adaptive", "enabled")
    if isinstance(last, list) and any(block.get("type") == "tool_result" for block in last):
        return {"text": "It is 18 degrees and overcast in Paris.", "stop": "end_turn"}
    if body.get("tools") and (body.get("tool_choice") or {}).get("type") == "any":
        return {"calls": [("toolu_any", {"city": "Paris"})], "stop": "tool_use"}
    if body.get("tools") and "Tokyo" in text:
        return {"calls": [("toolu_1", {"city": "Paris"}), ("toolu_2", {"city": "Tokyo"})], "stop": "tool_use"}
    if body.get("tools") and "weather" in text:
        reply: dict[str, Any] = {"calls": [("toolu_1", {"city": "Paris"})], "stop": "tool_use"}
        if thinking:
            reply["thinking"] = "a call"
        return reply
    if body["max_tokens"] == 8:
        return {"text": "The sea is", "stop": "max_tokens"}
    if _words(text) > 10_000:
        return {"text": "7741-ZEBRA", "stop": "end_turn"}
    if "9.11" in text:
        reply = {"text": "9.9", "stop": "end_turn"}
        if thinking:
            reply["thinking"] = "Compare the tenths."
        return reply
    return {"text": "ready", "stop": "end_turn"}


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

    def _error(self, status: int, kind: str, message: str, headers: dict[str, str] | None = None) -> None:
        self._json(status, {"type": "error", "error": {"type": kind, "message": message}, "request_id": None}, headers)

    def do_GET(self) -> None:
        if self.path.endswith("/models"):
            self._json(200, {"models": [{"name": "stub-model", "max_sequence_length": 131072}]})
            return
        self._error(404, "not_found_error", "not found")

    def _event(self, name: str, data: dict[str, Any]) -> None:
        self.wfile.write(f"event: {name}\ndata: {json.dumps({'type': name, **data})}\n\n".encode())

    def do_POST(self) -> None:
        body = json.loads(self.rfile.read(int(self.headers["content-length"])))
        if self.path.endswith("/count_tokens"):
            self._error(404, "not_found_error", "not found")
            return
        KNOBS["_posts"] = KNOBS.get("_posts", 0) + 1
        if KNOBS["_posts"] <= KNOBS.get("loading", 0):
            self._error(503, "overloaded_error", "loading", {"x-sie-error-code": "MODEL_LOADING", "retry-after": "1"})
            return
        if KNOBS.get("busy", 0) > len(KNOBS.setdefault("_busy", [])) and body.get("tools"):
            KNOBS["_busy"].append(body)
            self._error(503, "overloaded_error", "busy", {"x-sie-error-code": "RESOURCE_EXHAUSTED", "retry-after": "0"})
            return
        refusal = _refusal(body)
        if refusal:
            self._error(*refusal)
            return
        reply = _reply(body)
        usage = {
            "input_tokens": max(1, _words(_texts(body))),
            "cache_creation_input_tokens": 0,
            "cache_read_input_tokens": 0,
            "output_tokens": max(1, _words(reply.get("text", "")) + 3 * len(reply.get("calls", []))),
        }
        content: list[dict[str, Any]] = []
        if reply.get("thinking"):
            content.append({"type": "thinking", "thinking": reply["thinking"], "signature": "sie-unsigned"})
        if reply.get("text"):
            content.append({"type": "text", "text": reply["text"]})
        for call_id, arguments in reply.get("calls", []):
            content.append({"type": "tool_use", "id": call_id, "name": "get_weather", "input": arguments})
        if not body.get("stream"):
            self._json(
                200,
                {
                    "id": "msg_stub",
                    "type": "message",
                    "role": "assistant",
                    "model": body["model"],
                    "content": content,
                    "stop_reason": reply["stop"],
                    "stop_sequence": None,
                    "usage": usage,
                },
            )
            return
        self.send_response(200)
        self.send_header("content-type", "text/event-stream")
        self.send_header("x-sie-served-by", "stub")
        self.end_headers()
        zero = {key: 0 for key in usage}
        message = {"id": "msg_stub", "type": "message", "role": "assistant", "model": body["model"], "content": []}
        self._event("message_start", {"message": {**message, "stop_reason": None, "usage": zero}})
        self._event("ping", {})
        for index, block in enumerate(content):
            if KNOBS.get("delta_before_start") and index == 0:
                self._event("content_block_delta", {"index": 0, "delta": {"type": "text_delta", "text": "x"}})
            if block["type"] == "thinking":
                self._event("content_block_start", {"index": index, "content_block": {**block, "thinking": ""}})
                delta = {"type": "thinking_delta", "thinking": block["thinking"]}
                self._event("content_block_delta", {"index": index, "delta": delta})
            elif block["type"] == "text":
                self._event("content_block_start", {"index": index, "content_block": {"type": "text", "text": ""}})
                self._event(
                    "content_block_delta", {"index": index, "delta": {"type": "text_delta", "text": block["text"]}}
                )
            else:
                start = {**block, "input": {}}
                self._event("content_block_start", {"index": index, "content_block": start})
                delta = {"type": "input_json_delta", "partial_json": json.dumps(block["input"])}
                self._event("content_block_delta", {"index": index, "delta": delta})
            self._event("content_block_stop", {"index": index})
        delta_usage = {} if KNOBS.get("no_delta_usage") else {"usage": usage}
        self._event("message_delta", {"delta": {"stop_reason": reply["stop"], "stop_sequence": None}, **delta_usage})
        self._event("message_stop", {})


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
    report = conformance.run_messages(base_url, "", "stub-model")
    return {case["name"]: case for case in report["cases"]}


def test_a_conforming_endpoint_passes(base_url: str) -> None:
    cases = _cases(base_url)
    statuses = {name: case["status"] for name, case in cases.items()}
    recorded = {"tool_choice_any", "ping_events", "count_tokens"}
    assert {name for name, status in statuses.items() if status != "pass"} == recorded, statuses
    assert cases["streaming_tool_use"]["observed"]["tool_calls"][0]["first_delta_has_id_and_name"]
    assert cases["usage_on_message_delta"]["observed"]["on_message_delta"]
    assert cases["count_tokens"]["observed"]["status"] == 404
    assert cases["ping_events"]["observed"]["pings"] > 0
    assert cases["thinking_blocks"]["observed"]["without_thinking"]["reasoning_deltas"] == 0


def test_missing_usage_on_message_delta_fails(base_url: str) -> None:
    case = _cases(base_url, no_delta_usage=True)["usage_on_message_delta"]
    assert case["status"] == "fail"
    assert "no usage on message_delta" in case["detail"]


def test_a_delta_before_its_block_breaks_the_grammar(base_url: str) -> None:
    case = _cases(base_url, delta_before_start=True)["event_grammar"]
    assert case["status"] == "fail"
    assert "delta for block 0 while None is open" in case["detail"]


def test_an_accepted_effort_fails_the_refusal_case(base_url: str) -> None:
    assert _cases(base_url, effort_ok=True)["effort_refusal"]["status"] == "fail"


def test_a_system_turn_refusal_without_the_recovery_wording_fails(base_url: str) -> None:
    case = _cases(base_url, system_wording="messages.1: unexpected role")["system_turn_refusal"]
    assert case["status"] == "fail"
    assert "without the wording" in case["detail"]


def test_busy_refusals_are_retried(base_url: str) -> None:
    case = _cases(base_url, busy=2)["streaming_tool_use"]
    assert case["status"] == "pass"
    assert case["retries"] == 2


def test_the_warm_up_waits_out_a_loading_model(base_url: str) -> None:
    KNOBS.update(loading=2)
    report = conformance.run_messages(base_url, "", "stub-model")
    assert report["warm_up"]["attempts"] == 3
    assert report["summary"]["fail"] == 0


@pytest.mark.parametrize(
    ("events", "problem"),
    [
        ([{"event": "content_block_start", "index": 0}], "first event is content_block_start"),
        (
            [
                {"event": "message_start"},
                {"event": "content_block_start", "index": 1},
                {"event": "content_block_stop", "index": 1},
                {"event": "message_delta"},
                {"event": "message_stop"},
            ],
            "block index 1, expected 0",
        ),
        (
            [{"event": "message_start"}, {"event": "content_block_start", "index": 0}, {"event": "message_stop"}],
            "not message_delta then message_stop",
        ),
    ],
)
def test_grammar_problems(events: list[dict[str, Any]], problem: str) -> None:
    assert any(problem in found for found in conformance.grammar_problems(events))


def test_a_stream_ending_in_an_error_event_is_well_formed() -> None:
    events = [
        {"event": "message_start"},
        {"event": "ping"},
        {"event": "content_block_start", "index": 0},
        {"event": "content_block_delta", "index": 0},
        {"event": "error"},
    ]
    assert conformance.grammar_problems(events) == []
