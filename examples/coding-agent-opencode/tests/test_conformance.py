"""The conformance checker against a scripted OpenAI-compatible stub.

The stub plays the wire a conforming server sends, and each knob breaks one
thing, so every verdict is checked both for what it accepts and for what it
refuses. This is a check of the checker; the server itself is checked by
running ``conformance.py`` against a real endpoint.
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


def _reply(body: dict[str, Any]) -> dict[str, Any]:
    messages = body["messages"]
    tools = body.get("tools")
    user_text = "".join(m.get("content") or "" for m in messages if m["role"] == "user")
    if body.get("tool_choice") == "auto" and KNOBS.get("auto") == "refuse":
        return {"status": 400, "code": "unsupported_field", "message": "field 'tool_choice' is not supported"}
    if body.get("tool_choice") == "required":
        if KNOBS.get("required") == "refuse":
            return {"status": 400, "code": "unsupported_field", "message": "tool_choice 'required' is not supported"}
        if KNOBS.get("required") == "ignore":
            return {"content": "Hello!", "finish": "stop"}
        return {"calls": [("call_req", "get_weather", {"city": "Paris"})], "finish": "tool_calls"}
    if tools and messages[-1]["role"] == "tool":
        return {"content": "It is 18 degrees and overcast in Paris.", "finish": "stop"}
    if tools and "Tokyo" in user_text:
        calls = [("call_1", "get_weather", {"city": "Paris"}), ("call_2", "get_weather", {"city": "Tokyo"})]
        if KNOBS.get("parallel") == "single":
            calls = calls[:1]
        return {"calls": calls, "finish": "tool_calls"}
    if tools:
        return {"calls": [("call_1", "get_weather", {"city": "Paris"})], "finish": "tool_calls"}
    if body.get("max_tokens") == 8:
        return {"content": "The sea is", "finish": "length"}
    if len(user_text) > 100_000:
        if KNOBS.get("long") == "refuse":
            return {"status": 400, "code": "INPUT_TOO_LONG", "message": "prompt exceeds the context window"}
        return {"content": "7741-ZEBRA", "finish": "stop"}
    if "9.11" in user_text:
        reasoning = ""
        if body.get("include_reasoning") and KNOBS.get("reasoning", "yes") == "yes":
            reasoning = "0.9 is more than 0.11, so 9.9 is larger."
        if not body.get("include_reasoning") and KNOBS.get("reasoning_without_flag"):
            reasoning = "leaked"
        content = "<think>hmm</think>9.9" if KNOBS.get("leak_think") else "9.9"
        return {"content": content, "finish": "stop", "reasoning": reasoning}
    return {"content": "ready", "finish": "stop"}


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args: Any) -> None:
        pass

    def _json(self, status: int, payload: dict[str, Any]) -> None:
        raw = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def do_GET(self) -> None:
        if self.path.endswith("/models"):
            self._json(200, {"models": [{"name": "stub-model", "capabilities": {"tools": True}}]})
            return
        self._json(404, {"error": {"message": "no such route"}})

    def _chunk(self, choices: list[dict[str, Any]], **extra: Any) -> None:
        payload = {"id": "chatcmpl-stub", "object": "chat.completion.chunk", "created": 1, "model": "stub-model"}
        payload["choices"] = choices
        payload.update(extra)
        self.wfile.write(f"data: {json.dumps(payload)}\n\n".encode())

    def do_POST(self) -> None:
        body = json.loads(self.rfile.read(int(self.headers["content-length"])))
        KNOBS["_posts"] = KNOBS.get("_posts", 0) + 1
        if KNOBS["_posts"] <= KNOBS.get("loading", 0):
            self._json(
                503,
                {
                    "error": {
                        "message": "model is loading, please retry",
                        "type": "server_error",
                        "code": "MODEL_LOADING",
                    }
                },
            )
            return
        reply = _reply(body)
        if "status" in reply:
            self._json(
                reply["status"],
                {"error": {"message": reply["message"], "type": "invalid_request_error", "code": reply["code"]}},
            )
            return
        self.send_response(200)
        self.send_header("content-type", "text/event-stream")
        self.send_header("x-sie-served-by", "stub")
        self.end_headers()
        self._chunk([{"index": 0, "delta": {"role": "assistant", "content": ""}, "finish_reason": None}])
        for piece in (reply.get("reasoning") or "").split(" "):
            if piece:
                self._chunk([{"index": 0, "delta": {"reasoning_content": piece + " "}, "finish_reason": None}])
        for index, (call_id, name, arguments) in enumerate(reply.get("calls") or []):
            head = {"index": index, "type": "function", "function": {"name": name, "arguments": ""}}
            if not KNOBS.get("drop_ids"):
                head["id"] = call_id
            self._chunk([{"index": 0, "delta": {"tool_calls": [head]}, "finish_reason": None}])
            tail = {"index": index, "function": {"arguments": json.dumps(arguments)}}
            self._chunk([{"index": 0, "delta": {"tool_calls": [tail]}, "finish_reason": None}])
        for word in (reply.get("content") or "").split(" "):
            self._chunk([{"index": 0, "delta": {"content": word + " "}, "finish_reason": None}])
        self._chunk([{"index": 0, "delta": {}, "finish_reason": reply["finish"]}])
        if not KNOBS.get("omit_usage"):
            prompt_tokens = sum(len(m.get("content") or "") for m in body["messages"]) // 4
            usage = {"prompt_tokens": max(1, prompt_tokens), "completion_tokens": 3, "total_tokens": prompt_tokens + 3}
            self._chunk([], usage=usage)
        self.wfile.write(b"data: [DONE]\n\n")


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
def reset_knobs() -> Iterator[None]:
    KNOBS.clear()
    yield
    KNOBS.clear()


def _report(base_url: str, **knobs: Any) -> dict[str, Any]:
    KNOBS.update(knobs)
    return conformance.run(base_url, "", "stub-model")


def _run(base_url: str, **knobs: Any) -> dict[str, dict[str, Any]]:
    return {case["name"]: case for case in _report(base_url, **knobs)["cases"]}


def _failures(cases: dict[str, dict[str, Any]]) -> set[str]:
    return {name for name, case in cases.items() if case["status"] == "fail"}


def test_conforming_wire_passes_every_case(base_url: str) -> None:
    cases = _run(base_url)
    assert _failures(cases) == set()
    assert cases["tool_choice_auto"]["status"] == "pass"
    assert cases["tool_choice_required"]["status"] == "recorded"
    assert cases["tool_choice_required"]["observed"]["support"] == "supported"
    assert cases["prompt_above_64k_tokens"]["observed"]["needle_found"] is True
    assert cases["include_reasoning"]["observed"]["with_include_reasoning"]["reasoning_deltas"] > 0
    assert cases["streaming_tool_call_deltas"]["observed"]["headers"]["x-sie-served-by"] == "stub"


def test_tool_call_deltas_without_ids_fail(base_url: str) -> None:
    cases = _run(base_url, drop_ids=True)
    assert cases["streaming_tool_call_deltas"]["status"] == "fail"
    assert "no id" in cases["streaming_tool_call_deltas"]["detail"]
    assert cases["parallel_tool_calls"]["status"] == "fail"


def test_model_loading_is_waited_out(base_url: str, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(conformance, "WARM_UP_POLL_S", 0.01)
    report = _report(base_url, loading=3)
    assert report["warm_up"]["attempts"] == 4
    assert report["warm_up"]["error"] is None
    assert _failures({case["name"]: case for case in report["cases"]}) == set()


def test_tool_choice_auto_refused_fails_and_the_loop_is_still_checked(base_url: str) -> None:
    cases = _run(base_url, auto="refuse")
    assert cases["tool_choice_auto"]["status"] == "fail"
    assert "tool_choice" in cases["tool_choice_auto"]["detail"]
    assert cases["streaming_tool_call_deltas"]["status"] == "pass"
    assert cases["parallel_tool_calls"]["status"] == "pass"
    assert cases["tool_result_round_trip"]["status"] == "pass"


def test_one_call_for_two_cities_fails(base_url: str) -> None:
    cases = _run(base_url, parallel="single")
    assert cases["parallel_tool_calls"]["status"] == "fail"
    assert cases["parallel_tool_calls"]["detail"].startswith("1 tool call")


def test_required_refused_is_recorded_not_failed(base_url: str) -> None:
    cases = _run(base_url, required="refuse")
    assert cases["tool_choice_required"]["status"] == "recorded"
    assert cases["tool_choice_required"]["observed"]["support"] == "refused"
    assert cases["tool_choice_required"]["observed"]["error"]["status"] == 400


def test_required_accepted_but_ignored_fails(base_url: str) -> None:
    cases = _run(base_url, required="ignore")
    assert cases["tool_choice_required"]["status"] == "fail"


def test_missing_usage_fails(base_url: str) -> None:
    cases = _run(base_url, omit_usage=True)
    assert cases["usage_on_final_chunk"]["status"] == "fail"
    assert cases["prompt_above_64k_tokens"]["status"] == "fail"
    assert cases["finish_reason_stop"]["status"] == "pass"


def test_long_prompt_refused_fails_with_the_error_recorded(base_url: str) -> None:
    cases = _run(base_url, long="refuse")
    case = cases["prompt_above_64k_tokens"]
    assert case["status"] == "fail"
    assert case["observed"]["error"]["code"] == "INPUT_TOO_LONG"


def test_reasoning_field_rules(base_url: str) -> None:
    assert _run(base_url, reasoning="no")["include_reasoning"]["status"] == "pass"
    assert _run(base_url, leak_think=True)["include_reasoning"]["status"] == "fail"
    assert _run(base_url, reasoning_without_flag=True)["include_reasoning"]["status"] == "fail"
