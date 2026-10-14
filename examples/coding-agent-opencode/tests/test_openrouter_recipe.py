"""The OpenRouter recipe end to end, with a scripted upstream standing in for OpenRouter.

The repository's ``sie-server`` serves the recipe's models directory, with a
copy of the recipe's upstreams file pointed at a loopback stub. The stub
streams the way OpenRouter documents: SSE comment lines while it works,
reasoning in a ``reasoning`` field, tool-call deltas, and a usage event that
restates the finished choice. ``conformance.py`` then runs against the server,
as the README tells a reader to, and the stub records what the server sent.

A request whose ``provider`` block sets ``require_parameters`` gets a 404 from
the stub unless one of the model's OpenRouter endpoints lists every request
field other than the model, the messages, the stream flag and the provider
block. ``openrouter_endpoints.json`` holds each endpoint's
``supported_parameters`` from OpenRouter's public endpoints list.
"""

from __future__ import annotations

import json
import os
import socket
import subprocess
import threading
import time
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import httpx
import pytest

import conformance

EXAMPLE = Path(__file__).resolve().parents[1]
RECIPE = EXAMPLE / "recipes" / "openrouter"
REPOSITORY = EXAMPLE.parents[1]
SERVER = REPOSITORY / ".venv" / "bin" / "sie-server"
MODEL = "zai-org/GLM-5.3-Flash"
UPSTREAM_MODEL = "z-ai/glm-5.3-flash"
OPENROUTER_URL = "https://openrouter.ai/api/v1"
TEST_KEY = "sk-or-test-key"
SERVER_START_TIMEOUT_S = 300.0
REASONING = "Compare the tenths first: 9.9 has nine tenths and 9.11 has one tenth, so 9.9 is larger."
COMMENT = b": OPENROUTER PROCESSING\n\n"
ENDPOINTS = [
    frozenset(parameters)
    for parameters in json.loads(Path(__file__).with_name("openrouter_endpoints.json").read_text())[
        "supported_parameters"
    ]
]
REQUEST_FIELDS = frozenset({"model", "messages", "stream", "provider"})
NO_ENDPOINTS = "No endpoints found that can handle the requested parameters."

SEEN: list[dict[str, Any]] = []


def _unroutable(body: dict[str, Any]) -> bool:
    if not (body.get("provider") or {}).get("require_parameters"):
        return False
    parameters = set(body) - REQUEST_FIELDS
    return not any(parameters <= supported for supported in ENDPOINTS)


def _words(text: str) -> int:
    return len(text.split())


def _text(messages: list[dict[str, Any]], role: str) -> str:
    return " ".join(m["content"] for m in messages if m["role"] == role and isinstance(m.get("content"), str))


def _reply(body: dict[str, Any]) -> dict[str, Any]:
    messages = body["messages"]
    question = _text(messages, "user")
    tools = body.get("tools")
    paris = ("call_paris", "get_weather", {"city": "Paris"})
    if body.get("tool_choice") == "required":
        return {"calls": [paris], "finish": "tool_calls"}
    if tools and messages[-1]["role"] == "tool":
        return {"content": "It is 18 degrees and overcast in Paris.", "finish": "stop"}
    if tools and "Tokyo" in question:
        tokyo = ("call_tokyo", "get_weather", {"city": "Tokyo"})
        return {"calls": [paris, tokyo], "finish": "tool_calls", "reasoning": "Two cities, so two calls."}
    if tools:
        return {"calls": [paris], "finish": "tool_calls", "reasoning": "One call for Paris."}
    if body.get("max_tokens") == 8:
        return {"content": "The sea is", "finish": "length"}
    if _words(question) > 10_000:
        return {"content": "7741-ZEBRA", "finish": "stop"}
    if "9.11" in question:
        return {"content": "9.9", "finish": "stop", "reasoning": REASONING}
    return {"content": "ready", "finish": "stop"}


def _usage(body: dict[str, Any], reply: dict[str, Any]) -> dict[str, Any]:
    prompt = sum(_words(m["content"]) for m in body["messages"] if isinstance(m.get("content"), str))
    reasoning = _words(reply.get("reasoning") or "")
    arguments = " ".join(json.dumps(args) for _, _, args in reply.get("calls") or [])
    completion = max(1, _words(reply.get("content") or "") + reasoning + _words(arguments))
    return {
        "prompt_tokens": prompt,
        "completion_tokens": completion,
        "total_tokens": prompt + completion,
        "cost": 0.0001,
        "is_byok": False,
        "prompt_tokens_details": {"cached_tokens": 0},
        "completion_tokens_details": {"reasoning_tokens": reasoning},
    }


def _calls(reply: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {"index": index, "id": call_id, "type": "function", "function": {"name": name, "arguments": json.dumps(args)}}
        for index, (call_id, name, args) in enumerate(reply.get("calls") or [])
    ]


class OpenRouterStub(BaseHTTPRequestHandler):
    def log_message(self, *args: Any) -> None:
        pass

    def _event(self, choices: list[dict[str, Any]], **extra: Any) -> None:
        payload = {
            "id": "gen-stub",
            "provider": "Stub",
            "model": UPSTREAM_MODEL,
            "object": "chat.completion.chunk",
            "created": 1,
            "choices": choices,
            **extra,
        }
        self.wfile.write(f"data: {json.dumps(payload)}\n\n".encode() + COMMENT)

    def _delta(self, delta: dict[str, Any], finish: str | None = None) -> dict[str, Any]:
        return {"index": 0, "delta": {"role": "assistant", "content": "", **delta}, "finish_reason": finish}

    def _json(self, status: int, payload: dict[str, Any]) -> None:
        raw = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def do_POST(self) -> None:
        body = json.loads(self.rfile.read(int(self.headers["content-length"])))
        SEEN.append({"path": self.path, "authorization": self.headers.get("authorization"), "body": body})
        if _unroutable(body):
            self._json(404, {"error": {"message": NO_ENDPOINTS, "code": 404}})
            return
        reply = _reply(body)
        usage = _usage(body, reply)
        if not body.get("stream"):
            message: dict[str, Any] = {"role": "assistant", "content": reply.get("content") or "", "refusal": None}
            if reply.get("reasoning"):
                message["reasoning"] = reply["reasoning"]
                message["reasoning_details"] = [{"type": "reasoning.text", "text": reply["reasoning"]}]
            if reply.get("calls"):
                message["tool_calls"] = _calls(reply)
            choice = {"index": 0, "message": message, "finish_reason": reply["finish"]}
            self._json(200, {"id": "gen-stub", "model": UPSTREAM_MODEL, "choices": [choice], "usage": usage})
            return
        self.send_response(200)
        self.send_header("content-type", "text/event-stream")
        self.end_headers()
        self.wfile.write(COMMENT + COMMENT)
        self._event([self._delta({})])
        for piece in (reply.get("reasoning") or "").split(" "):
            if piece:
                details = [{"type": "reasoning.text", "text": piece + " "}]
                self._event([self._delta({"reasoning": piece + " ", "reasoning_details": details})])
        for call in _calls(reply):
            head = {**call, "function": {"name": call["function"]["name"], "arguments": ""}}
            self._event([self._delta({"tool_calls": [head]})])
            tail = {"index": call["index"], "function": {"arguments": call["function"]["arguments"]}}
            self._event([self._delta({"tool_calls": [tail]})])
        for word in (reply.get("content") or "").split(" "):
            if word:
                self._event([self._delta({"content": word + " "})])
        finished = {**self._delta({}, reply["finish"]), "native_finish_reason": reply["finish"]}
        self._event([finished])
        self._event([finished], usage=usage)
        self.wfile.write(b"data: [DONE]\n\n")


def _free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


def _wait_for_health(base: str, server: subprocess.Popen[bytes], log: Path) -> None:
    deadline = time.monotonic() + SERVER_START_TIMEOUT_S
    while time.monotonic() < deadline:
        if server.poll() is not None:
            pytest.fail(f"sie-server exited {server.returncode}:\n{log.read_text()[-4000:]}")
        try:
            if httpx.get(f"{base}/healthz", timeout=2.0).status_code == 200:
                return
        except httpx.HTTPError:
            pass
        time.sleep(0.5)
    pytest.fail(f"sie-server did not answer /healthz within {SERVER_START_TIMEOUT_S}s:\n{log.read_text()[-4000:]}")


@pytest.fixture(scope="module")
def report(tmp_path_factory: pytest.TempPathFactory) -> Iterator[dict[str, Any]]:
    if not SERVER.exists():
        message = f"{SERVER} is missing; run `mise run sync` at the repository root"
        if os.environ.get("CI"):
            pytest.fail(message)
        pytest.skip(message)
    stub = ThreadingHTTPServer(("127.0.0.1", 0), OpenRouterStub)
    threading.Thread(target=stub.serve_forever, daemon=True).start()
    work = tmp_path_factory.mktemp("openrouter")
    recipe = (RECIPE / "upstreams.yaml").read_text()
    assert recipe.count(OPENROUTER_URL) == 1
    upstreams = work / "upstreams.yaml"
    upstreams.write_text(recipe.replace(OPENROUTER_URL, f"http://127.0.0.1:{stub.server_address[1]}/api/v1"))
    port = _free_port()
    env = {key: value for key, value in os.environ.items() if key not in ("VIRTUAL_ENV", "SIE_REMOTE_SERVING")}
    env["OPENROUTER_API_KEY"] = TEST_KEY
    log = work / "sie-server.log"
    command = [
        str(SERVER),
        "serve",
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--device",
        "cpu",
        "--models-dir",
        str(RECIPE / "models"),
        "--upstreams-file",
        str(upstreams),
    ]
    SEEN.clear()
    with log.open("wb") as output:
        server = subprocess.Popen(command, env=env, stdout=output, stderr=subprocess.STDOUT)
    try:
        base = f"http://127.0.0.1:{port}"
        _wait_for_health(base, server, log)
        yield conformance.run(f"{base}/v1", "", MODEL)
    finally:
        server.terminate()
        try:
            server.wait(timeout=30)
        except subprocess.TimeoutExpired:
            server.kill()
            server.wait()
        stub.shutdown()


def _cases(report: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {case["name"]: case for case in report["cases"]}


def test_every_case_passes_through_sie(report: dict[str, Any]) -> None:
    cases = _cases(report)
    assert {name for name, case in cases.items() if case["status"] != "pass"} == {"tool_choice_required"}, cases
    assert cases["tool_choice_required"]["status"] == "recorded"
    assert cases["tool_choice_required"]["observed"]["support"] == "supported"
    assert report["served_by"] == "remote"
    assert report["model_entry"]["max_sequence_length"] == 131072


def test_reasoning_reaches_the_client_only_when_asked(report: dict[str, Any]) -> None:
    observed = _cases(report)["include_reasoning"]["observed"]
    assert observed["model_reasoned"] is True
    assert observed["with_include_reasoning"]["reasoning_deltas"] > 0
    assert observed["without_include_reasoning"]["reasoning_deltas"] == 0
    assert observed["non_streamed_with_include_reasoning"]["reasoning_chars"] == len(REASONING)
    assert report["tools_turn_params"] == {"tool_choice": "auto", "extra_body": {"include_reasoning": True}}


def test_the_upstream_receives_the_recipes_request(report: dict[str, Any]) -> None:
    assert report["cases"]
    assert SEEN
    every_endpoint = frozenset.intersection(*ENDPOINTS)
    for request in SEEN:
        body = request["body"]
        assert request["path"] == "/api/v1/chat/completions"
        assert request["authorization"] == f"Bearer {TEST_KEY}"
        assert body["model"] == UPSTREAM_MODEL
        fields = {"model", "messages", "stream", "max_tokens", "temperature"}
        if body["stream"]:
            fields.add("stream_options")
        if "tools" in body:
            fields |= {"tools", "tool_choice"}
        assert set(body) == fields, body
        assert fields - REQUEST_FIELDS - {"stream_options"} <= every_endpoint
    assert {request["body"]["stream"] for request in SEEN} == {True, False}
    round_trip = [r["body"] for r in SEEN if r["body"]["messages"][-1]["role"] == "tool"]
    assert round_trip
    assert round_trip[0]["messages"][-2]["reasoning_content"] == "One call for Paris. "


def test_the_stand_in_routes_require_parameters_by_the_endpoint_lists() -> None:
    body = {
        "model": UPSTREAM_MODEL,
        "messages": [{"role": "user", "content": "hi"}],
        "stream": True,
        "max_tokens": 4,
        "temperature": 0,
        "tools": [],
        "tool_choice": "auto",
    }
    strict = {**body, "provider": {"require_parameters": True}}
    assert len(ENDPOINTS) == 33
    assert not _unroutable(strict)
    assert not _unroutable({**strict, "parallel_tool_calls": True})
    assert _unroutable({**strict, "parallel_tool_calls": True, "top_k": 20})
    assert _unroutable({**strict, "n": 1})
    assert _unroutable({**strict, "stream_options": {"include_usage": True}})
    assert not _unroutable({**body, "stream_options": {"include_usage": True}, "n": 1})
