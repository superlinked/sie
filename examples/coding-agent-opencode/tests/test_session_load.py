"""The session load generator against a scripted OpenAI-compatible stub.

The stub counts one prompt token per word, streams a fixed answer one word per
chunk and reports a cached prefix when a request repeats an earlier one. That
is enough to check what the generator sends, how it grows each transcript and
how it summarises the turns; capacity is measured by running it against a
real endpoint.
"""

from __future__ import annotations

import json
import sys
import threading
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import pytest

import session_load

ANSWER = " ".join(f"word{i}" for i in range(40))
REQUESTS: list[dict[str, Any]] = []
KNOBS: dict[str, Any] = {}
LOCK = threading.Lock()


def _prompt_words(body: dict[str, Any]) -> list[str]:
    return [word for message in body["messages"] for word in (message.get("content") or "").split()]


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args: Any) -> None:
        pass

    def _send(self, payload: dict[str, Any]) -> None:
        self.wfile.write(f"data: {json.dumps(payload)}\n\n".encode())

    def do_GET(self) -> None:
        raw = json.dumps({"models": [{"name": "stub-model", "max_output_tokens": 32768}]}).encode()
        self.send_response(200)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def do_POST(self) -> None:
        body = json.loads(self.rfile.read(int(self.headers["content-length"])))
        words = _prompt_words(body)
        with LOCK:
            cached = max((_shared(words, earlier) for earlier in REQUESTS), default=0)
            REQUESTS.append({"body": body, "words": words})
            failing = KNOBS.get("fail_turn") and f"turn {KNOBS['fail_turn']}." in body["messages"][-1]["content"]
        if failing:
            raw = json.dumps({"error": {"message": "worker failed", "code": "INTERNAL_ERROR"}}).encode()
            self.send_response(500)
            self.send_header("content-type", "application/json")
            self.send_header("content-length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)
            return
        answer = ANSWER.split()[: body["max_tokens"]]
        self.send_response(200)
        self.send_header("content-type", "text/event-stream")
        self.end_headers()
        base = {"id": "chatcmpl-stub", "object": "chat.completion.chunk", "created": 1, "model": "stub-model"}
        for word in answer:
            self._send({**base, "choices": [{"index": 0, "delta": {"content": word + " "}, "finish_reason": None}]})
        self._send({**base, "choices": [{"index": 0, "delta": {}, "finish_reason": "length"}]})
        usage = {
            "prompt_tokens": len(words),
            "completion_tokens": len(answer),
            "total_tokens": len(words) + len(answer),
            "prompt_tokens_details": {"cached_tokens": cached},
        }
        self._send({**base, "choices": [], "usage": usage})
        self.wfile.write(b"data: [DONE]\n\n")


def _shared(words: list[str], earlier: dict[str, Any]) -> int:
    count = 0
    for mine, theirs in zip(words, earlier["words"], strict=False):
        if mine != theirs:
            break
        count += 1
    return count


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
def reset() -> Iterator[None]:
    REQUESTS.clear()
    KNOBS.clear()
    yield
    REQUESTS.clear()
    KNOBS.clear()


def _main(base_url: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *extra: str) -> tuple[int, dict[str, Any]]:
    report = tmp_path / "load.json"
    argv = ["session_load.py", "--base-url", base_url, "--model", "stub-model", "--report", str(report)]
    small = ["--prefix-tokens", "400", "--turn-tokens", "120", "--turns", "3", "--output-tokens", "20"]
    monkeypatch.setattr(sys, "argv", [*argv, *small, *extra])
    code = session_load.main()
    return code, json.loads(report.read_text())


def test_each_session_replays_its_whole_transcript(
    base_url: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    code, _ = _main(base_url, tmp_path, monkeypatch, "--sessions", "2")
    assert code == 0
    turns = [request["body"] for request in REQUESTS if len(request["body"]["messages"]) > 2]
    third = [body for body in turns if "turn 3." in body["messages"][-1]["content"]]
    assert len(third) == 2
    for body in third:
        roles = [message["role"] for message in body["messages"]]
        assert roles == ["system", "user", "assistant", "user", "assistant", "user"]
        assert body["messages"][2]["content"].split() == ANSWER.split()[:20]
        assert body["stream"] is True
        assert body["max_tokens"] == 20
        assert "temperature" not in body
    assert third[0]["messages"][0] == third[1]["messages"][0]
    assert third[0]["messages"][1] != third[1]["messages"][1]


def test_the_input_is_sized_with_the_endpoints_token_count(
    base_url: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, report = _main(base_url, tmp_path, monkeypatch, "--sessions", "1")
    assert report["shape"]["tokens_per_word"] == pytest.approx(1.0, abs=0.1)
    first = next(turn for turn in report["turns"] if turn["turn"] == 1)
    assert 400 + 120 <= first["prompt_tokens"] <= 400 + 120 + 80


def test_the_summary_counts_turns_and_cached_prefix(
    base_url: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    code, report = _main(base_url, tmp_path, monkeypatch, "--sessions", "1,3")
    assert code == 0
    one, three = report["levels"]
    assert (one["sessions"], one["turns"], one["errors"]) == (1, 3, 0)
    assert (three["sessions"], three["turns"], three["errors"]) == (3, 9, 0)
    assert 0.5 < three["returning_cached_share"] < 1.0
    assert three["max_prompt_tokens"] > one["max_prompt_tokens"] * 0.9
    for level in report["levels"]:
        assert level["decode_tps_p50"] > 0
        assert level["returning_ttft_p90_s"] is not None
    assert len(report["turns"]) == 12


def test_a_failed_turn_ends_its_session_and_fails_the_run(
    base_url: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    KNOBS["fail_turn"] = 2
    code, report = _main(base_url, tmp_path, monkeypatch, "--sessions", "2")
    assert code == 1
    (level,) = report["levels"]
    assert (level["turns"], level["errors"], level["meets_bar"]) == (4, 2, False)
    failed = [turn for turn in report["turns"] if turn["error"]]
    assert {turn["turn"] for turn in failed} == {2}
    assert all("INTERNAL_ERROR" in turn["error"] or "500" in turn["error"] for turn in failed)


def test_the_bar_is_judged_at_p90() -> None:
    def turn(number: int, ttft: float, tps: float) -> session_load.Turn:
        return session_load.Turn(sessions=1, session=1, turn=number, ttft_s=ttft, decode_tps=tps, prompt_tokens=10)

    fast = [turn(1, 9.0, 60.0)] + [turn(n, 0.5, 60.0) for n in range(2, 12)]
    assert session_load.summarise(1, fast, 1.0)["meets_bar"] is True
    slow_ttft = fast[:-2] + [turn(11, 4.0, 60.0), turn(12, 4.0, 60.0)]
    assert session_load.summarise(1, slow_ttft, 1.0)["meets_bar"] is False
    slow_decode = fast[:-2] + [turn(11, 0.5, 20.0), turn(12, 0.5, 20.0)]
    assert session_load.summarise(1, slow_decode, 1.0)["meets_bar"] is False
