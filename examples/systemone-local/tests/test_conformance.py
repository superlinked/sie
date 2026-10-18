"""The conformance checker against a stub System One server.

The stub answers as SIE's ``POST /v1/systemone`` and ``GET /v1/models`` do,
and each knob breaks one thing, so every case is checked both for what it
accepts and for what it refuses. This checks the checker, offline; SIE itself
is checked by running ``conformance.py`` against a running server.
"""

from __future__ import annotations

import json
import threading
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import pytest

import conformance

MODEL = "acme/decide"
WINDOW_WORDS = 400
KNOBS: dict[str, Any] = {}


def _answer(question: dict[str, Any]) -> dict[str, Any]:
    if question["type"] == "noul":
        return {"type": "noul", "noul": 0.25, "answer": False, "confidence": 0.75}
    if question["type"] == "choice":
        keys = list(question["criteria"])
        rest = 0.2 / max(1, len(keys) - 1)
        probabilities = {key: (0.8 if i == 0 else rest) for i, key in enumerate(keys)}
        return {"type": "choice", "choice": keys[0], "probabilities": probabilities, "confidence": 0.7}
    levels = question["criteria"]
    probabilities = {str(i): 1.0 / len(levels) for i in range(len(levels))}
    return {
        "type": "score",
        "score": (len(levels) - 1) / 2,
        "legend": {str(i): level for i, level in enumerate(levels)},
        "probabilities": probabilities,
        "confidence": 0.0,
    }


def _invalid(message: str) -> tuple[int, dict[str, Any]]:
    return 422, {"detail": [{"type": "value_error", "loc": ["body"], "msg": message}]}


def _systemone(body: dict[str, Any], authorization: str | None) -> tuple[int, dict[str, Any]]:
    if KNOBS.get("check_key") and authorization != "Bearer good-key":
        return 401, {"detail": {"error_type": "authentication_error", "message": "Invalid token"}}
    if body.get("model") != MODEL:
        return 404, {"detail": {"error_type": "not_found_error", "message": f"Model {body.get('model')!r} not found"}}
    questions = body.get("questions") or {}
    for question in questions.values():
        if question.get("type") not in {"noul", "choice", "score"}:
            return _invalid("unknown question type")
        if question["type"] == "score" and len(question.get("criteria", [])) > 10 and not KNOBS.get("any_levels"):
            return _invalid("at most 10 levels")
    state = body.get("state")
    if isinstance(state, str) and len(state.split()) > WINDOW_WORDS and not KNOBS.get("truncate"):
        return _invalid("the state does not fit the model's window")
    answers = {qid: _answer(question) for qid, question in questions.items()}
    if KNOBS.get("drop_answer"):
        answers.popitem()
    tokens: Any = KNOBS.get("input_tokens", 12.0 if KNOBS.get("float_tokens") else 12)
    return 200, {"model": MODEL, "answers": answers, "usage": {"input_tokens": tokens, "output_tokens": 0}}


def _models() -> dict[str, Any]:
    entry: dict[str, Any] = {"name": MODEL, "description": "A stub decision model.", "release_date": "2026-09-24"}
    if KNOBS.get("no_release_date"):
        del entry["release_date"]
    return {"models": [entry, {"name": "acme/embedder", "description": "", "release_date": ""}]}


class Handler(BaseHTTPRequestHandler):
    def _send(self, status: int, body: dict[str, Any]) -> None:
        data = json.dumps(body).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self) -> None:
        if self.path == "/v1/models":
            self._send(200, _models())
        else:
            self._send(404, {"detail": "Not Found"})

    def do_POST(self) -> None:
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        if self.path == "/v1/systemone":
            self._send(*_systemone(body, self.headers.get("Authorization")))
        else:
            self._send(404, {"detail": "Not Found"})

    def log_message(self, *_: Any) -> None:
        pass


@pytest.fixture
def server(monkeypatch: pytest.MonkeyPatch) -> Iterator[str]:
    KNOBS.clear()
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    url = f"http://127.0.0.1:{httpd.server_address[1]}"
    monkeypatch.setenv("TYPESAFE_BASE_URL", url)
    monkeypatch.setenv("TYPESAFE_DEFAULT_MODEL", MODEL)
    monkeypatch.setenv("TYPESAFE_API_KEY", "good-key")
    yield url
    httpd.shutdown()


def _verdicts(**kwargs: Any) -> dict[str, bool]:
    return {result.name: result.passed for result in conformance.run(**kwargs)}


def test_conforming_server_passes_every_case(server: str) -> None:
    KNOBS["check_key"] = True
    verdicts = _verdicts(check_auth=True)
    assert all(verdicts.values()), [name for name, passed in verdicts.items() if not passed]
    assert len(verdicts) == len(conformance.CASES) + 1


@pytest.mark.parametrize(
    ("knob", "failing"),
    [
        ("truncate", {"state_too_long"}),
        ("any_levels", {"too_many_levels"}),
        ("no_release_date", {"models_list"}),
        ("float_tokens", None),
        ("drop_answer", None),
    ],
)
def test_each_break_is_caught(server: str, knob: str, failing: set[str] | None) -> None:
    KNOBS[knob] = True
    verdicts = _verdicts()
    failed = {name for name, passed in verdicts.items() if not passed}
    if failing is None:
        # A body the SDK cannot parse, or a missing answer, fails every answering case.
        assert {"noul", "choice", "score", "envelope"} & failed
    else:
        assert failed == failing


def test_report_lists_every_case(server: str) -> None:
    results = conformance.run()
    text = conformance.report(results)
    assert text.splitlines()[-1] == f"{len(results)}/{len(results)} passed"


@pytest.mark.parametrize("gateway", [False, True])
@pytest.mark.parametrize("tokens", [None, -1, 0, 12])
def test_envelope_requires_explicit_gateway_for_zero_usage(server: str, gateway: bool, tokens: int | None) -> None:
    KNOBS["input_tokens"] = tokens
    assert _verdicts(gateway=gateway)["envelope"] is (tokens is not None and tokens >= (0 if gateway else 1))


def test_gateway_flag_is_explicit_and_recorded(server: str, tmp_path: Path) -> None:
    KNOBS["input_tokens"] = 0
    assert conformance.main([]) == 1
    output = tmp_path / "gateway.json"
    assert conformance.main(["--gateway", "--json", str(output)]) == 0
    payload = json.loads(output.read_text())
    assert payload["gateway"] is True
    assert all(case["passed"] for case in payload["cases"])
