"""The OpenRouter recipe end to end, with a scripted upstream standing in for OpenRouter.

The repository's ``sie-server`` serves the recipe's models directory, with a
copy of the recipe's upstreams file pointed at a loopback stub. The stub
streams the way OpenRouter documents: SSE comment lines while it works,
reasoning in a ``reasoning`` field, tool-call deltas, and a usage event that
restates the finished choice. ``conformance.py`` then runs against the server,
as the README tells a reader to, and the stub records what the server sent.

OpenRouter documents ``require_parameters`` as routing only to providers that
support every LLM parameter in the request, and names those parameters on its
parameters page. The stub applies that rule to the parameter names OpenRouter's
public model list uses, and answers 404 when no endpoint of the model lists
every such name in the request. Other fields, such as ``stream_options``, do not
count. ``openrouter_endpoints.json`` holds each endpoint's
``supported_parameters`` from OpenRouter's public endpoints list, and the
parameter names from its public model list.

A request whose only user message is one of ``REFUSALS`` or ``ERROR_EVENT``
gets that failure instead of an answer, the way OpenRouter sends it.
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
from types import SimpleNamespace
from typing import Any

import httpx
import openai
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
PUBLISHED = json.loads(Path(__file__).with_name("openrouter_endpoints.json").read_text())
ENDPOINTS = [frozenset(parameters) for parameters in PUBLISHED["supported_parameters"]]
PARAMETERS = frozenset(PUBLISHED["catalog_parameters"])
NO_ENDPOINTS = "No endpoints found that can handle the requested parameters."
UPSTREAM_DETAIL = "stand-in upstream detail"
REFUSALS = {"answer 401": 401, "answer 402": 402, "answer 404": 404}
ERROR_EVENT = "answer with an error event"

SEEN: list[dict[str, Any]] = []


def _unroutable(body: dict[str, Any]) -> bool:
    if not (body.get("provider") or {}).get("require_parameters"):
        return False
    parameters = set(body) & PARAMETERS
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
    if tools and tools[0]["function"]["name"] == "write_file":
        lines = range(1, conformance.LONG_CALL_LINES + 1)
        content = "\n".join(conformance.long_call_line(number) for number in lines)
        return {
            "calls": [("call_file", "write_file", {"path": conformance.LONG_CALL_PATH, "content": content})],
            "finish": "tool_calls",
        }
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
        question = _text(body["messages"], "user")
        if question in REFUSALS:
            self._json(REFUSALS[question], {"error": {"message": UPSTREAM_DETAIL, "code": REFUSALS[question]}})
            return
        if question == ERROR_EVENT:
            self.send_response(200)
            self.send_header("content-type", "text/event-stream")
            self.end_headers()
            self.wfile.write(COMMENT)
            error = {"code": "server_error", "message": UPSTREAM_DETAIL}
            self._event([{"index": 0, "delta": {"content": ""}, "finish_reason": "error"}], error=error)
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
            arguments = call["function"]["arguments"]
            for start in range(0, len(arguments), 4096):
                tail = {"index": call["index"], "function": {"arguments": arguments[start : start + 4096]}}
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
def server(tmp_path_factory: pytest.TempPathFactory) -> Iterator[SimpleNamespace]:
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
        yield SimpleNamespace(base=base, log=log)
    finally:
        server.terminate()
        try:
            server.wait(timeout=30)
        except subprocess.TimeoutExpired:
            server.kill()
            server.wait()
        stub.shutdown()


@pytest.fixture(scope="module")
def report(server: SimpleNamespace) -> dict[str, Any]:
    return conformance.run(f"{server.base}/v1", "", MODEL)


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


def _conformance_requests() -> list[dict[str, Any]]:
    return [r for r in SEEN if _text(r["body"]["messages"], "user") not in {*REFUSALS, ERROR_EVENT}]


def test_each_case_reaches_the_upstream_with_exactly_these_fields(report: dict[str, Any]) -> None:
    assert report["cases"]
    ready = "Reply with the single word: ready."
    paris = "What is the weather in Paris right now? Use the get_weather tool."
    cities = "Get the current weather in Paris and in Tokyo. Call get_weather once per city, both in this turn."
    long = f"Read this warehouse log, then answer.\n\n{conformance.long_prompt()}\n\nWhat is the access code? Reply with just the code."
    budget = conformance.GENERATION_BUDGET
    plain = {"model": UPSTREAM_MODEL, "stream": False, "temperature": 0, "max_tokens": budget}
    streamed = {"model": UPSTREAM_MODEL, "stream": True, "stream_options": {"include_usage": True}, "temperature": 0}
    tools = {**streamed, "max_tokens": budget, "tools": [conformance.WEATHER_TOOL], "tool_choice": "auto"}
    expected = [
        (ready, ["user"], {**streamed, "max_tokens": 4}),
        (conformance.REASONING_QUESTION, ["user"], {**streamed, "max_tokens": budget}),
        (conformance.REASONING_QUESTION, ["user"], {**streamed, "max_tokens": budget}),
        (conformance.REASONING_QUESTION, ["user"], plain),
        (paris, ["user"], tools),
        (cities, ["user"], tools),
        (paris, ["user", "assistant", "tool"], tools),
        ("the long prompt", ["user"], {**streamed, "max_tokens": budget}),
        (ready, ["user"], {**streamed, "max_tokens": budget}),
        ("Write three paragraphs about the sea.", ["user"], {**streamed, "max_tokens": 8}),
        ("Say hello.", ["user"], {**tools, "tool_choice": "required"}),
        (
            conformance.LONG_CALL_PROMPT,
            ["user"],
            {**tools, "max_tokens": conformance.LONG_CALL_BUDGET, "tools": [conformance.WRITE_TOOL]},
        ),
    ]
    received = _conformance_requests()
    sent = []
    for request in received:
        body = request["body"]
        prompt = body["messages"][0]["content"]
        roles = [message["role"] for message in body["messages"]]
        fields = {name: value for name, value in body.items() if name != "messages"}
        sent.append(("the long prompt" if prompt == long else prompt, roles, fields))
        assert request["path"] == "/api/v1/chat/completions"
        assert request["authorization"] == f"Bearer {TEST_KEY}"
        assert not _unroutable({**body, "provider": {"require_parameters": True}})
    assert sent == expected
    assert received[6]["body"]["messages"][1]["reasoning_content"] == "One call for Paris. "


def test_the_stand_in_counts_only_openrouter_parameters_for_require_parameters() -> None:
    body = {
        "model": UPSTREAM_MODEL,
        "messages": [{"role": "user", "content": "hi"}],
        "stream": True,
        "stream_options": {"include_usage": True},
        "max_tokens": 4,
        "temperature": 0,
        "tools": [],
        "tool_choice": "auto",
    }
    strict = {**body, "provider": {"require_parameters": True}}
    assert len(ENDPOINTS) == 33
    assert len(PARAMETERS) == 26
    assert not _unroutable(strict)
    assert not _unroutable({**strict, "n": 1})
    assert not _unroutable({**strict, "parallel_tool_calls": True})
    assert _unroutable({**strict, "parallel_tool_calls": True, "top_k": 20})
    assert _unroutable({**strict, "max_completion_tokens": 8})
    assert not _unroutable({**body, "max_completion_tokens": 8, "parallel_tool_calls": True, "top_k": 20})


@pytest.mark.parametrize(
    ("question", "reason", "message"),
    [
        ("answer 401", "upstream answered 401", "the upstream rejected the server's credential"),
        ("answer 402", "upstream answered 402", "the upstream account cannot pay for the request"),
        ("answer 404", "upstream answered 404", "the upstream has no route for the configured model"),
        (
            ERROR_EVENT,
            "upstream sent an error instead of a chat answer",
            "the upstream could not be reached, timed out or failed",
        ),
    ],
)
def test_a_refusal_answers_502_and_the_server_log_names_why(
    server: SimpleNamespace, question: str, reason: str, message: str
) -> None:
    client = openai.OpenAI(base_url=f"{server.base}/v1", api_key="none", max_retries=0)
    result = conformance.stream_chat(client, MODEL, [{"role": "user", "content": question}], max_tokens=16)
    assert result.error is not None
    assert (result.error["status"], result.error["code"], result.error["message"]) == (502, "upstream_error", message)
    log = server.log.read_text()
    assert f"remote chat for {MODEL} failed: {reason}" in log
    assert UPSTREAM_DETAIL not in log
