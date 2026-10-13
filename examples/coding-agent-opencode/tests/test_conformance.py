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
SEEN: dict[str, list[dict[str, Any]]] = {}

REASONING = " ".join(["Compare the tenths first: 9.9 has nine tenths and 9.11 has one tenth."] * 5)


def _words(text: str) -> int:
    return len(text.split())


def _reply(body: dict[str, Any]) -> dict[str, Any]:
    messages = body["messages"]
    tools = body.get("tools")
    user_text = "".join(m.get("content") or "" for m in messages if m["role"] == "user")
    if body.get("include_reasoning") and KNOBS.get("reasoning") == "refuse":
        return {"status": 400, "code": "unsupported_field", "message": "field 'include_reasoning' is not supported"}
    if body.get("tool_choice") == "auto" and KNOBS.get("auto") == "refuse":
        return {"status": 400, "code": "unsupported_field", "message": "field 'tool_choice' is not supported"}
    if body.get("tool_choice") == "required":
        if KNOBS.get("required") == "refuse":
            return {"status": 400, "code": "unsupported_field", "message": "tool_choice 'required' is not supported"}
        if KNOBS.get("required") == "rate_limited":
            return {"status": 429, "code": "rate_limit_exceeded", "message": "rate limit exceeded, retry later"}
        if KNOBS.get("required") == "ignore":
            return {"content": "Hello!", "finish": "stop"}
        return {"calls": [("call_req", "get_weather", {"city": "Paris"})], "finish": "tool_calls"}
    if tools and messages[-1]["role"] == "tool":
        SEEN.setdefault("round_trip_assistant", []).append(messages[-2])
        return {"content": "It is 18 degrees and overcast in Paris.", "finish": "stop"}
    if tools:
        SEEN.setdefault("tools_turns", []).append(body)
        if KNOBS.get("tools_error"):
            return {"status": 500, "code": "INTERNAL_ERROR", "message": "worker failed"}
        if KNOBS.get("busy", 0) > len(SEEN.get("busy", [])):
            SEEN.setdefault("busy", []).append(body)
            return {"status": 503, "code": "QUEUE_FULL", "message": "upstream busy"}
        if KNOBS.get("midstream_error"):
            return {"midstream_error": True}
    if tools and "Tokyo" in user_text:
        calls = [("call_1", "get_weather", {"city": "Paris"}), ("call_2", "get_weather", {"city": "Tokyo"})]
        if KNOBS.get("parallel") == "single":
            calls = calls[:1]
        return {"calls": calls, "finish": "tool_calls", "reasoning": "two cities, two calls"}
    if tools:
        return {"calls": [("call_1", "get_weather", {"city": "Paris"})], "finish": "tool_calls", "reasoning": "a call"}
    if body.get("max_tokens") == 8:
        return {"content": "The sea is", "finish": "length"}
    if _words(user_text) > 10_000:
        if KNOBS.get("long") == "refuse":
            return {"status": 400, "code": "INPUT_TOO_LONG", "message": "prompt exceeds the context window"}
        return {"content": "7741-ZEBRA", "finish": "stop"}
    if "9.11" in user_text:
        mode = KNOBS.get("reasoning", "yes")
        reasoning = REASONING if mode in ("yes", "ignored", "stream_only") else ""
        if mode == "stream_only" and not body.get("stream"):
            return {"content": "9.9", "finish": "stop", "hidden": reasoning}
        if KNOBS.get("reasoning_without_flag") and not body.get("include_reasoning"):
            return {"content": "9.9", "finish": "stop", "reasoning": "leaked"}
        if not body.get("include_reasoning") or mode == "ignored":
            return {"content": "9.9", "finish": "stop", "hidden": reasoning}
        content = "<think>hmm</think>9.9" if KNOBS.get("leak_think") else "9.9"
        return {"content": content, "finish": "stop", "reasoning": reasoning}
    return {"content": "ready", "finish": "stop"}


def _thinking_budget_spent(body: dict[str, Any], reply: dict[str, Any]) -> dict[str, Any]:
    think = KNOBS.get("think_tokens", 0)
    max_tokens = body.get("max_tokens")
    if not think or max_tokens in (4, 8) or "status" in reply or reply.get("midstream_error"):
        return reply
    if max_tokens is None or max_tokens < think + 64:
        return {"content": "", "finish": "length", "hidden": "think " * (max_tokens or 0)}
    return reply


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

    def do_GET(self) -> None:
        if self.path.endswith("/models"):
            if KNOBS.get("models_list"):
                self._json(200, [{"id": "stub-model"}])
                return
            entry: dict[str, Any] = {"name": "stub-model", "capabilities": {"tools": True}}
            if KNOBS.get("window"):
                entry["max_sequence_length"] = KNOBS["window"]
            self._json(200, {"models": [entry]})
            return
        self._json(404, {"error": {"message": "no such route"}})

    def _chunk(self, choices: list[dict[str, Any]], **extra: Any) -> None:
        payload = {"id": "chatcmpl-stub", "object": "chat.completion.chunk", "created": 1, "model": "stub-model"}
        payload["choices"] = choices
        payload.update(extra)
        self.wfile.write(f"data: {json.dumps(payload)}\n\n".encode())

    def _usage(self, body: dict[str, Any], reply: dict[str, Any]) -> dict[str, int]:
        prompt_tokens = sum(_words(m.get("content") or "") for m in body["messages"])
        completion = " ".join(
            [reply.get("content") or "", reply.get("reasoning") or "", reply.get("hidden") or ""]
            + [json.dumps(arguments) for _, _, arguments in reply.get("calls") or []]
        )
        completion_tokens = max(1, _words(completion))
        return {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        }

    def do_POST(self) -> None:
        body = json.loads(self.rfile.read(int(self.headers["content-length"])))
        KNOBS["_posts"] = KNOBS.get("_posts", 0) + 1
        if KNOBS["_posts"] <= KNOBS.get("loading", 0):
            error = {"message": "model is loading, please retry", "type": "server_error", "code": "MODEL_LOADING"}
            self._json(503, {"error": error})
            return
        reply = _thinking_budget_spent(body, _reply(body))
        if "status" in reply:
            error = {"message": reply["message"], "type": "invalid_request_error", "code": reply["code"]}
            retry = {"retry-after": str(KNOBS.get("retry_after", 0))} if reply["status"] in (429, 503) else {}
            self._json(reply["status"], {"error": error}, retry)
            return
        if not body.get("stream"):
            message: dict[str, Any] = {"role": "assistant", "content": reply.get("content")}
            if reply.get("reasoning"):
                message["reasoning_content"] = reply["reasoning"]
            payload = {
                "id": "chatcmpl-stub",
                "object": "chat.completion",
                "created": 1,
                "model": "stub-model",
                "choices": [{"index": 0, "message": message, "finish_reason": reply["finish"]}],
                "usage": self._usage(body, reply),
            }
            self._json(200, payload)
            return
        self.send_response(200)
        self.send_header("content-type", "text/event-stream")
        self.send_header("x-sie-served-by", "stub")
        self.send_header("x-sie-upstream", "vendor-name-from-the-profile")
        self.end_headers()
        self._chunk([{"index": 0, "delta": {"role": "assistant", "content": ""}, "finish_reason": None}])
        if reply.get("midstream_error"):
            self._chunk(
                [{"index": 0, "delta": {}, "finish_reason": None}],
                error={"message": "upstream busy", "type": "server_error", "param": None, "code": "QUEUE_FULL"},
                request_id="req-stub",
            )
            self.wfile.write(b"data: [DONE]\n\n")
            return
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
            if word:
                self._chunk([{"index": 0, "delta": {"content": word + " "}, "finish_reason": None}])
        self._chunk([{"index": 0, "delta": {}, "finish_reason": reply["finish"]}])
        if not KNOBS.get("omit_usage"):
            self._chunk([], usage=self._usage(body, reply))
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
    SEEN.clear()
    yield
    KNOBS.clear()
    SEEN.clear()


def _report(base_url: str, **knobs: Any) -> dict[str, Any]:
    KNOBS.update(knobs)
    return conformance.run(base_url, "", "stub-model")


def _run(base_url: str, **knobs: Any) -> dict[str, dict[str, Any]]:
    return {case["name"]: case for case in _report(base_url, **knobs)["cases"]}


def _failures(cases: dict[str, dict[str, Any]]) -> set[str]:
    return {name for name, case in cases.items() if case["status"] == "fail"}


def test_conforming_wire_passes_every_case(base_url: str) -> None:
    report = _report(base_url)
    cases = {case["name"]: case for case in report["cases"]}
    assert _failures(cases) == set()
    assert cases["tool_choice_auto"]["status"] == "pass"
    assert cases["tool_choice_required"]["status"] == "recorded"
    assert cases["tool_choice_required"]["observed"]["support"] == "supported"
    assert cases["prompt_above_64k_tokens"]["observed"]["needle_found"] is True
    reasoning = cases["include_reasoning"]["observed"]
    assert reasoning["model_reasoned"] is True
    assert reasoning["with_include_reasoning"]["reasoning_deltas"] > 0
    assert reasoning["non_streamed_with_include_reasoning"]["reasoning_chars"] == len(REASONING)
    assert cases["streaming_tool_call_deltas"]["observed"]["headers"] == {"x-sie-served-by": "stub"}
    assert report["served_by"] == "stub"
    assert "vendor-name-from-the-profile" not in json.dumps(report)


def test_tools_turns_carry_include_reasoning_and_send_reasoning_back(base_url: str) -> None:
    report = _report(base_url)
    assert report["tools_turn_params"] == {"tool_choice": "auto", "extra_body": {"include_reasoning": True}}
    assert SEEN["tools_turns"]
    assert all(turn.get("include_reasoning") is True for turn in SEEN["tools_turns"])
    (assistant,) = SEEN["round_trip_assistant"]
    assert assistant["content"] == ""
    assert assistant["reasoning_content"] == "a call "
    assert assistant["tool_calls"][0]["id"] == "call_1"


def test_refused_include_reasoning_fails_and_the_tools_turns_go_without_it(base_url: str) -> None:
    report = _report(base_url, reasoning="refuse")
    cases = {case["name"]: case for case in report["cases"]}
    assert cases["include_reasoning"]["status"] == "fail"
    assert "include_reasoning refused: 400" in cases["include_reasoning"]["detail"]
    assert report["tools_turn_params"] == {"tool_choice": "auto"}
    assert all("include_reasoning" not in turn for turn in SEEN["tools_turns"])
    assert _failures(cases) == {"include_reasoning"}


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


def test_busy_refusals_are_retried_and_recorded(base_url: str) -> None:
    cases = _run(base_url, busy=2)
    assert _failures(cases) == set()
    assert cases["tool_choice_auto"]["status"] == "pass"
    retries = cases["streaming_tool_call_deltas"]["observed"]["retries"]
    assert [(retry["status"], retry["code"]) for retry in retries] == [(503, "QUEUE_FULL"), (503, "QUEUE_FULL")]


def test_a_failed_tools_turn_does_not_blame_tool_choice(base_url: str) -> None:
    cases = _run(base_url, tools_error=True)
    assert cases["tool_choice_auto"]["status"] == "recorded"
    assert cases["tool_choice_auto"]["detail"].startswith("not judged")
    assert cases["streaming_tool_call_deltas"]["status"] == "fail"


def test_an_error_event_mid_stream_is_recorded_not_raised(base_url: str) -> None:
    cases = _run(base_url, midstream_error=True)
    case = cases["streaming_tool_call_deltas"]
    assert case["status"] == "fail"
    assert case["observed"]["error"]["code"] == "QUEUE_FULL"
    assert case["observed"]["error"]["message"] == "upstream busy"
    assert cases["finish_reason_stop"]["status"] == "pass"


def test_a_models_list_answered_as_a_json_array_keeps_the_report(base_url: str) -> None:
    report = _report(base_url, models_list=True)
    assert report["model_entry"] == {"id": "stub-model"}
    assert _failures({case["name"]: case for case in report["cases"]}) == set()


def test_tool_choice_auto_refused_fails_and_the_loop_is_still_checked(base_url: str) -> None:
    report = _report(base_url, auto="refuse")
    cases = {case["name"]: case for case in report["cases"]}
    assert cases["tool_choice_auto"]["status"] == "fail"
    assert "tool_choice" in cases["tool_choice_auto"]["detail"]
    assert cases["streaming_tool_call_deltas"]["status"] == "pass"
    assert cases["parallel_tool_calls"]["status"] == "pass"
    assert cases["tool_result_round_trip"]["status"] == "pass"
    assert report["tools_turn_params"] == {"extra_body": {"include_reasoning": True}}


def test_one_call_for_two_cities_fails(base_url: str) -> None:
    cases = _run(base_url, parallel="single")
    assert cases["parallel_tool_calls"]["status"] == "fail"
    assert cases["parallel_tool_calls"]["detail"].startswith("1 tool call")


def test_required_refused_is_recorded_not_failed(base_url: str) -> None:
    cases = _run(base_url, required="refuse")
    assert cases["tool_choice_required"]["status"] == "recorded"
    assert cases["tool_choice_required"]["observed"]["support"] == "refused"
    assert cases["tool_choice_required"]["observed"]["error"]["status"] == 400


def test_required_rate_limited_is_a_failure_not_a_refusal(base_url: str, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(conformance, "RETRY_BUDGET_S", 0.2)
    cases = _run(base_url, required="rate_limited", retry_after=0.05)
    case = cases["tool_choice_required"]
    assert case["status"] == "fail"
    assert "support" not in case["observed"]
    assert case["observed"]["error"]["status"] == 429
    assert case["observed"]["retries"]


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


def test_long_prompt_under_a_smaller_declared_window_is_recorded(base_url: str) -> None:
    refused = _run(base_url, long="refuse", window=16384)["prompt_above_64k_tokens"]
    assert refused["status"] == "recorded"
    assert refused["detail"].startswith("declared window 16384 is under 65536; refused: 400 INPUT_TOO_LONG")
    KNOBS.clear()
    accepted = _run(base_url, window=8192)["prompt_above_64k_tokens"]
    assert accepted["status"] == "recorded"
    assert "yet" in accepted["detail"]


def test_long_prompt_clears_the_threshold_one_token_per_word() -> None:
    assert len(conformance.long_prompt().split()) >= conformance.LONG_PROMPT_MIN_TOKENS * 1.03


def test_a_thinking_model_has_room_to_answer_every_case(base_url: str) -> None:
    cases = _run(base_url, think_tokens=3000)
    assert _failures(cases) == set()
    assert cases["tool_choice_required"]["observed"]["support"] == "supported"


def test_reasoning_field_rules(base_url: str) -> None:
    assert _run(base_url, reasoning="no")["include_reasoning"]["status"] == "pass"
    assert _run(base_url, leak_think=True)["include_reasoning"]["status"] == "fail"
    assert _run(base_url, reasoning_without_flag=True)["include_reasoning"]["status"] == "fail"


def test_a_reasoning_model_whose_server_drops_the_flag_fails(base_url: str) -> None:
    case = _run(base_url, reasoning="ignored")["include_reasoning"]
    assert case["status"] == "fail"
    assert "no delta.reasoning_content in the stream" in case["detail"]
    assert "no message.reasoning_content" in case["detail"]


def test_reasoning_missing_from_the_non_streamed_message_fails(base_url: str) -> None:
    case = _run(base_url, reasoning="stream_only")["include_reasoning"]
    assert case["status"] == "fail"
    assert "no message.reasoning_content" in case["detail"]
    assert "in the stream" not in case["detail"]
