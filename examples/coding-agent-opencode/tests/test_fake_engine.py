"""``conformance.py`` against the repository's ``sie-server`` serving the weightless fake.

The server runs the fake bundle on the CPU, and the check runs against its
``sie-fake:chat`` profile the way the README runs it against a real model. The
fake answers from a script: with tools declared it calls the first one, passing
the last user message as every required argument, after a ``role: tool``
message it answers with the tool result, and it reasons in one sentence first.
So each case here checks SIE's single-node chat wire, not a model.

The fake makes one call per turn, so ``parallel_tool_calls`` fails, and the
test asserts that failure instead of skipping the case. Its 4,096-token output
cap refuses the 32,000 tokens a 64 KiB call needs, so ``long_tool_call`` is
recorded as not run. The fake does not read
the long prompt, so the needle is never found; the case passes on the prompt
size and the answer, as it does for any model.

The same server answers ``conformance.py --api messages``: ``/v1/messages``
on ``sie-server`` translates each request onto the chat route above and the
answer back, so the Messages cases check that translation over the same
scripted turns.
"""

from __future__ import annotations

import json
import os
import socket
import subprocess
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import httpx
import pytest

import conformance

EXAMPLE = Path(__file__).resolve().parents[1]
REPOSITORY = EXAMPLE.parents[1]
SERVER = REPOSITORY / ".venv" / "bin" / "sie-server"
MODEL = "sie-fake:chat"
WINDOW = 131072
OUTPUT_CAP = 4096
SERVER_START_TIMEOUT_S = 300.0
TOOL_RESULT = {"city": "Paris", "temperature_c": 18, "conditions": "overcast"}


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
def server(tmp_path_factory: pytest.TempPathFactory) -> Iterator[str]:
    if not SERVER.exists():
        message = f"{SERVER} is missing; run `mise run sync` at the repository root"
        if os.environ.get("CI"):
            pytest.fail(message)
        pytest.skip(message)
    port = _free_port()
    env = {key: value for key, value in os.environ.items() if key != "VIRTUAL_ENV"}
    log = tmp_path_factory.mktemp("fake-engine") / "sie-server.log"
    command = [str(SERVER), "serve", "--host", "127.0.0.1", "--port", str(port), "--device", "cpu", "-b", "fake"]
    with log.open("wb") as output:
        process = subprocess.Popen(command, env=env, stdout=output, stderr=subprocess.STDOUT)
    try:
        base = f"http://127.0.0.1:{port}"
        _wait_for_health(base, process, log)
        yield base
    finally:
        process.terminate()
        try:
            process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()


@pytest.fixture(scope="module")
def report(server: str) -> dict[str, Any]:
    return conformance.run(f"{server}/v1", "", MODEL)


def _cases(report: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {case["name"]: case for case in report["cases"]}


def test_each_case_against_the_fake(report: dict[str, Any]) -> None:
    statuses = {name: case["status"] for name, case in _cases(report).items()}
    assert statuses == {
        "tool_choice_auto": "pass",
        "streaming_tool_call_deltas": "pass",
        "parallel_tool_calls": "fail",
        "tool_result_round_trip": "pass",
        "prompt_above_64k_tokens": "pass",
        "finish_reason_stop": "pass",
        "finish_reason_tool_calls": "pass",
        "finish_reason_length": "pass",
        "usage_on_final_chunk": "pass",
        "include_reasoning": "pass",
        "tool_choice_required": "recorded",
        "long_tool_call": "recorded",
    }, report["cases"]
    assert report["served_by"] == "local"
    assert report["warm_up"]["error"] is None
    assert report["model_entry"]["max_sequence_length"] == WINDOW
    assert report["tools_turn_params"] == {"tool_choice": "auto", "extra_body": {"include_reasoning": True}}


def test_the_fake_makes_one_call_per_turn(report: dict[str, Any]) -> None:
    case = _cases(report)["parallel_tool_calls"]
    assert case["detail"].startswith("1 tool call in the turn: get_weather("), case
    assert [call["name"] for call in case["observed"]["tool_calls"]] == ["get_weather"]


def test_the_tool_call_streams_its_id_and_name_first(report: dict[str, Any]) -> None:
    observed = _cases(report)["streaming_tool_call_deltas"]["observed"]
    [call] = observed["tool_calls"]
    question = "What is the weather in Paris right now? Use the get_weather tool."
    assert call["first_delta_has_id_and_name"] is True
    assert call["deltas"] > 1
    assert json.loads(call["arguments"]) == {"city": question}
    assert observed["finish_reason"] == "tool_calls"
    assert observed["reasoning_deltas"] > 0


def test_the_tool_result_reaches_the_answer(report: dict[str, Any]) -> None:
    observed = _cases(report)["tool_result_round_trip"]["observed"]
    assert observed["content"] == f"Tool result: {json.dumps(TOOL_RESULT)}"
    assert observed["tool_calls"] == []
    assert observed["reasoning_content_sent_back"]


def test_reasoning_comes_back_only_when_asked(report: dict[str, Any]) -> None:
    observed = _cases(report)["include_reasoning"]["observed"]
    with_flag = observed["with_include_reasoning"]
    without_flag = observed["without_include_reasoning"]
    assert observed["model_reasoned"] is True
    assert with_flag["reasoning_deltas"] > 0
    assert without_flag["reasoning_deltas"] == 0
    assert observed["non_streamed_with_include_reasoning"]["reasoning_chars"] == with_flag["reasoning_chars"]
    assert with_flag["usage"] == without_flag["usage"]


def test_the_long_prompt_fits_the_declared_window(report: dict[str, Any]) -> None:
    observed = _cases(report)["prompt_above_64k_tokens"]["observed"]
    prompt_tokens = observed["usage"]["prompt_tokens"]
    assert conformance.LONG_PROMPT_MIN_TOKENS < prompt_tokens < WINDOW
    assert observed["declared_window"] == WINDOW
    assert observed["needle_found"] is False
    assert observed["finish_reason"] == "stop"


def test_max_tokens_and_usage_are_exact(report: dict[str, Any]) -> None:
    cases = _cases(report)
    length = cases["finish_reason_length"]["observed"]
    assert length["usage"]["completion_tokens"] == 8
    usage = cases["usage_on_final_chunk"]["observed"]["usage"]
    assert usage["total_tokens"] == usage["prompt_tokens"] + usage["completion_tokens"]


def test_the_long_tool_call_is_not_sent_past_the_declared_output_cap(report: dict[str, Any]) -> None:
    case = _cases(report)["long_tool_call"]
    assert case["observed"]["error"]["param"] == "max_tokens"
    assert case["detail"].startswith("not run: max_tokens=32000 refused: 400 INVALID_INPUT")
    assert f"model cap ({OUTPUT_CAP})" in case["detail"]


def test_tool_choice_required_forces_a_call(report: dict[str, Any]) -> None:
    observed = _cases(report)["tool_choice_required"]["observed"]
    assert observed["support"] == "supported"
    assert json.loads(observed["tool_calls"][0]["arguments"]) == {"city": "Say hello."}


@pytest.fixture(scope="module")
def messages_report(server: str) -> dict[str, Any]:
    return conformance.run_messages(f"{server}/v1", "", MODEL)


def test_each_messages_case_against_the_fake(messages_report: dict[str, Any]) -> None:
    statuses = {name: case["status"] for name, case in _cases(messages_report).items()}
    assert statuses == {
        "streaming_tool_use": "pass",
        "parallel_tool_use": "fail",
        "tool_result_round_trip": "pass",
        "prompt_above_64k_tokens": "pass",
        "stop_reason_end_turn": "pass",
        "stop_reason_tool_use": "pass",
        "stop_reason_max_tokens": "pass",
        "usage_on_message_delta": "pass",
        "thinking_blocks": "pass",
        "tool_choice_any": "recorded",
        "buffered_message": "pass",
        "inert_fields_accepted": "pass",
        "effort_refusal": "pass",
        "system_turn_refusal": "pass",
        "unknown_model": "pass",
        "event_grammar": "pass",
        "ping_events": "recorded",
        "count_tokens": "recorded",
    }, messages_report["cases"]
    assert messages_report["served_by"] == "local"
    assert messages_report["warm_up"]["error"] is None


def test_the_tool_use_block_streams_the_scripted_call(messages_report: dict[str, Any]) -> None:
    observed = _cases(messages_report)["streaming_tool_use"]["observed"]
    [call] = observed["tool_calls"]
    question = "What is the weather in Paris right now? Use the get_weather tool."
    assert call["first_delta_has_id_and_name"] is True
    assert json.loads(call["arguments"]) == {"city": question}
    assert observed["stop_reason"] == "tool_use"
    assert observed["reasoning_deltas"] > 0


def test_messages_usage_is_the_chat_usage_translated(messages_report: dict[str, Any]) -> None:
    cases = _cases(messages_report)
    usage = cases["usage_on_message_delta"]["observed"]["usage"]
    long_prompt = cases["prompt_above_64k_tokens"]["observed"]["usage"]
    assert usage["input_tokens"] > 0
    assert usage["output_tokens"] > 0
    assert usage["cache_creation_input_tokens"] == 0
    assert cases["usage_on_message_delta"]["observed"]["message_start_usage"]["input_tokens"] == 0
    assert conformance.LONG_PROMPT_MIN_TOKENS < long_prompt["input_tokens"] < WINDOW
    assert cases["stop_reason_max_tokens"]["observed"]["usage"]["output_tokens"] == 8


def test_thinking_blocks_come_back_only_when_asked(messages_report: dict[str, Any]) -> None:
    observed = _cases(messages_report)["thinking_blocks"]["observed"]
    assert observed["with_thinking"]["reasoning_deltas"] > 0
    assert observed["without_thinking"]["reasoning_deltas"] == 0
    assert observed["non_streamed_with_thinking"]["reasoning_chars"] == observed["with_thinking"]["reasoning_chars"]


def test_the_messages_refusals_use_the_anthropic_wording(messages_report: dict[str, Any]) -> None:
    cases = _cases(messages_report)
    assert cases["effort_refusal"]["detail"] == (
        "400 invalid_request_error: output_config.effort: Extra inputs are not permitted"
    )
    assert cases["unknown_model"]["observed"]["error"]["type"] == "not_found_error"
    assert cases["count_tokens"]["observed"]["status"] == 404
    assert cases["count_tokens"]["observed"]["body"]["error"]["type"] == "not_found_error"
