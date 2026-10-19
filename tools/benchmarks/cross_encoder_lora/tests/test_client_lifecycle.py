"""Wire evidence survives without the benchmark retaining successful responses."""

from __future__ import annotations

import asyncio
import gc
import json
import weakref
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import httpx
import pytest
from sie_sdk._msgpack import packb, unpackb
from sie_sdk.client.errors import ServerError, SIEConnectionError

from tools.benchmarks.cross_encoder_lora import client as worker
from tools.benchmarks.cross_encoder_lora.evidence import Evidence

HTTP_CLIENT = httpx.Client
URL = "http://client-lifecycle.invalid"
TIMING = {
    "x-queue-time": "1.25",
    "x-tokenization-time": "0.02",
    "x-inference-time": "3.75",
    "x-postprocessing-time": "0.01",
    "x-total-time": "5.03",
}


def success(request: httpx.Request) -> httpx.Response:
    body = unpackb(request.content, numeric_arrays=False)
    return httpx.Response(
        200,
        content=packb(
            {
                "model": worker.MODEL,
                "scores": [
                    {"item_id": item["id"], "score": float(4 - index), "rank": index}
                    for index, item in enumerate(body["items"])
                ],
                "usage": {"input_tokens": 17},
            }
        ),
        headers={"content-type": "application/msgpack", **TIMING},
    )


def install_transport(
    monkeypatch: pytest.MonkeyPatch, handler: Callable[[httpx.Request], httpx.Response]
) -> list[weakref.ReferenceType[httpx.Client]]:
    clients = []

    def factory(*args: Any, **kwargs: Any) -> httpx.Client:
        client = HTTP_CLIENT(*args, **kwargs, transport=httpx.MockTransport(handler))
        clients.append(weakref.ref(client))
        return client

    monkeypatch.setattr(worker.httpx, "Client", factory)
    return clients


class TrackedStream(httpx.SyncByteStream):
    def __init__(self, body: bytes) -> None:
        self.body = body
        self.reads = 0

    def __iter__(self) -> Iterator[bytes]:
        self.reads += 1
        yield self.body


def test_success_hook_snapshots_only_primitive_status_headers_without_reading_body() -> None:
    client = object.__new__(worker.Client)
    client.responses = []
    stream = TrackedStream(b"body must remain unread by hook")
    response = httpx.Response(200, headers={**TIMING, "unrelated": "ignored"}, stream=stream)
    client.observe(response)
    assert stream.reads == 0
    assert client.responses == [{"status": 200, "timing_headers": TIMING}]
    assert json.loads(json.dumps(client.responses)) == client.responses
    response.headers["x-total-time"] = "changed"
    assert client.responses[0]["timing_headers"]["x-total-time"] == "5.03"


def test_actual_sdk_success_history_does_not_retain_response_or_closed_owner(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    responses = []

    def handler(request: httpx.Request) -> httpx.Response:
        response = success(request)
        responses.append(weakref.ref(response))
        return response

    install_transport(monkeypatch, handler)
    client = worker.Client(URL)
    rows = [client.score(None, f"request-{i}", timed=True) for i in range(32)]
    for row in rows:
        worker.validate_response(row)
    before = json.dumps(rows, sort_keys=True)
    assert all(row["responses"] == [{"status": 200, "timing_headers": TIMING}] for row in rows)
    # Ordinary HTTPX response-stream cycles remain. They can be collected while
    # this live benchmark Client still owns all of its primitive wire history.
    gc.collect()
    assert len(responses) == 32
    assert all(ref() is None for ref in responses)
    sdk, http = client.sdk, client._http
    request_hooks = list(http.event_hooks["request"])

    def unrelated(_: httpx.Response) -> None:
        pass

    http.event_hooks["response"].append(unrelated)
    owner = weakref.ref(client)
    client.close()
    assert http.is_closed
    assert not sdk._finalizer.alive
    assert http.event_hooks["response"] == [unrelated]
    assert http.event_hooks["request"] == request_hooks
    assert client.responses == []
    assert json.dumps(rows, sort_keys=True) == before
    del client
    # SDK/http are intentionally still live: the removed benchmark hook must
    # no longer keep its owner alive. No library-owned cycle is modified.
    assert owner() is None
    assert sdk.jobs._c is sdk


@pytest.mark.parametrize("encoding", ["json", "msgpack", "malformed"])
def test_error_body_and_retry_evidence_are_finalized_outside_hook(encoding: str) -> None:
    body = {"error": {"code": "LORA_LOADING", "message": "adapter is loading"}}
    content = json.dumps(body).encode() if encoding == "json" else packb(body)
    if encoding == "malformed":
        content = b"not JSON"
    stream = TrackedStream(content)
    response = httpx.Response(
        503,
        headers={
            "content-type": "application/msgpack" if encoding == "msgpack" else "application/json",
            "retry-after": "0.2",
            **TIMING,
        },
        stream=stream,
    )
    client = object.__new__(worker.Client)
    client.responses = []
    client.observe(response)
    assert stream.reads == 0
    assert client.responses[0] is response
    row = {"responses": client.responses.copy()}
    worker.serialize_responses(row)
    evidence = row["responses"][0]
    assert stream.reads == 1
    assert evidence["status"] == 503
    assert evidence["timing_headers"] == TIMING
    assert evidence["body_hex"] == content.hex()
    assert evidence["retry_after"] == "0.2"
    if encoding == "malformed":
        assert evidence["body_decode_error"] == "JSONDecodeError"
        assert evidence["code"] is None
    else:
        assert evidence["code"] == "LORA_LOADING"
    json.dumps(row)


@pytest.mark.parametrize(("code", "cold"), [("LORA_LOADING", True), ("MODEL_LOADING", False)])
def test_actual_sdk_attempts_preserve_cold_retry_and_reject_hidden_retry(
    monkeypatch: pytest.MonkeyPatch, code: str, cold: bool
) -> None:
    attempts = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            return httpx.Response(
                503,
                content=packb({"error": {"code": code, "message": "loading"}}),
                headers={"content-type": "application/msgpack", "retry-after": "0", **TIMING},
            )
        return success(request)

    install_transport(monkeypatch, handler)
    client = worker.Client(URL)
    try:
        row = client.score("a" if cold else None, "retry", cold=cold, timed=True)
        if cold:
            worker.validate_response(row, allow_loading=True)
        else:
            with pytest.raises(AssertionError, match="hidden SDK retry"):
                worker.validate_response(row)
        assert attempts == 2
        assert [value["status"] for value in row["responses"]] == [503, 200]
        assert row["responses"][0]["code"] == code
        assert row["responses"][0]["retry_after"] == "0"
        assert row["responses"][1]["timing_headers"] == TIMING
        saved = json.dumps(row, sort_keys=True)
    finally:
        client.close()
    assert json.dumps(row, sort_keys=True) == saved


def test_partial_performance_failure_preserves_unreturned_wire_attempt_before_close(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    attempts = 0
    closed = []
    original_close = worker.Client.close

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        if attempts == 5:
            return httpx.Response(
                500,
                content=packb({"error": {"code": "INTERNAL_ERROR", "message": "injected failure"}}),
                headers={"content-type": "application/msgpack", **TIMING},
            )
        return success(request)

    def close(client: worker.Client) -> None:
        original_close(client)
        closed.append(client._http.is_closed and not client.responses)

    install_transport(monkeypatch, handler)
    monkeypatch.setattr(worker.Client, "close", close)
    evidence = Evidence(tmp_path / "evidence")
    with pytest.raises(ServerError):
        worker.performance(URL, 1, evidence, smoke=True)
    record = json.loads((evidence.directory / "events.jsonl").read_text())
    assert record["event"] == "performance_attempt"
    assert len(record["warmups"]) == 2
    assert len(record["completed_samples"]) == 2
    assert [row["status"] for row in record["wire_responses"]] == [200, 200, 200, 200, 500]
    failure = record["wire_responses"][-1]
    assert failure["code"] == "INTERNAL_ERROR"
    assert unpackb(bytes.fromhex(failure["body_hex"]), numeric_arrays=False)["error"]["message"] == "injected failure"
    assert closed == [True]


def test_owned_hooks_and_history_clear_even_if_sdk_close_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    install_transport(monkeypatch, success)
    client = worker.Client(URL)
    row = client.score(None, "close-failure")
    saved = json.dumps(row, sort_keys=True)
    http = client._http

    def unrelated(_: httpx.Response) -> None:
        pass

    def fail() -> None:
        raise OSError("injected close failure")

    http.event_hooks["response"].append(unrelated)
    monkeypatch.setattr(client.sdk, "close", fail)
    try:
        with pytest.raises(OSError, match="injected close failure"):
            client.close()
        assert http.event_hooks["response"] == [unrelated]
        assert client.responses == []
        assert json.dumps(row, sort_keys=True) == saved
    finally:
        http.close()


def test_partially_read_error_stream_preserves_original_sdk_failure_and_all_attempts(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    attempts = 0
    closed = []
    original_close = worker.Client.close

    class BrokenStream(httpx.SyncByteStream):
        def __iter__(self) -> Iterator[bytes]:
            yield b'{"error":'
            raise httpx.ReadError("injected partial error body")

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        if attempts == 5:
            return httpx.Response(
                500,
                stream=BrokenStream(),
                headers={"content-type": "application/json", "retry-after": "0.2", **TIMING},
            )
        return success(request)

    def close(client: worker.Client) -> None:
        original_close(client)
        closed.append(client._http.is_closed and not client.responses)

    install_transport(monkeypatch, handler)
    monkeypatch.setattr(worker.Client, "close", close)
    evidence = Evidence(tmp_path / "evidence")
    with pytest.raises(SIEConnectionError, match="Connection lost mid-request") as failure:
        worker.performance(URL, 1, evidence, smoke=True)
    assert isinstance(failure.value.__cause__, httpx.ReadError)
    record = json.loads((evidence.directory / "events.jsonl").read_text())
    assert len(record["warmups"]) == 2
    assert len(record["completed_samples"]) == 2
    assert [row["status"] for row in record["wire_responses"]] == [200, 200, 200, 200, 500]
    final = record["wire_responses"][-1]
    assert final["timing_headers"] == TIMING
    assert final["content_type"] == "application/json"
    assert final["retry_after"] == "0.2"
    assert final["body_unavailable"] is True
    assert final["body_read_error"] == "StreamConsumed"
    assert final["code"] is None
    assert "body_hex" not in final
    assert closed == [True]


@pytest.mark.parametrize("body_failure", ["transport", "decoder"])
def test_unavailable_error_body_retains_metadata_for_transport_and_decoder_failures(body_failure: str) -> None:
    class BrokenStream(httpx.SyncByteStream):
        def __iter__(self) -> Iterator[bytes]:
            raise httpx.ReadError("injected stream failure")
            yield b""  # pragma: no cover

    headers = {"content-type": "application/json", "retry-after": "0.2", **TIMING}
    if body_failure == "decoder":
        headers["content-encoding"] = "gzip"
        stream = httpx.ByteStream(b"this is not gzip")
    else:
        stream = BrokenStream()
    response = httpx.Response(500, headers=headers, stream=stream)
    record = worker.describe_response(response)
    assert record["status"] == 500
    assert record["timing_headers"] == TIMING
    assert record["content_type"] == "application/json"
    assert record["retry_after"] == "0.2"
    assert record["body_unavailable"] is True
    assert record["body_read_error"] == ("DecodingError" if body_failure == "decoder" else "ReadError")
    assert "body_hex" not in record


@pytest.mark.parametrize("cancel_type", [KeyboardInterrupt, SystemExit, asyncio.CancelledError])
def test_deferred_error_body_does_not_swallow_cancellation(cancel_type: type[BaseException]) -> None:
    class CancelledStream(httpx.SyncByteStream):
        def __iter__(self) -> Iterator[bytes]:
            raise cancel_type("cancelled")
            yield b""  # pragma: no cover

    response = httpx.Response(500, stream=CancelledStream())
    with pytest.raises(cancel_type):
        worker.describe_response(response)


def test_performance_evidence_failure_still_closes_all_clients(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    closed = []
    original_close = worker.Client.close
    install_transport(monkeypatch, success)

    def close(client: worker.Client) -> None:
        original_close(client)
        closed.append(client._http.is_closed and not client.responses)

    def emit(*_: Any, **__: Any) -> None:
        raise OSError("injected evidence failure")

    monkeypatch.setattr(worker.Client, "close", close)
    evidence = Evidence(tmp_path / "evidence")
    monkeypatch.setattr(evidence, "emit", emit)
    gc_enabled = gc.isenabled()
    with pytest.raises(OSError, match="injected evidence failure"):
        worker.performance(URL, 4, evidence, smoke=True)
    assert closed == [True] * 4
    assert gc.isenabled() == gc_enabled


def test_primary_sdk_error_survives_observer_failure_with_all_wire_attempts(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    attempts = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        if attempts == 4:  # two warmups, one completed timed request, one failure
            return httpx.Response(500, json={"error": {"code": "INTERNAL_ERROR", "message": "primary SDK failure"}})
        return success(request)

    def after(_clock: dict[str, float]) -> None:
        raise BrokenPipeError("secondary observer failed")

    install_transport(monkeypatch, handler)
    evidence = Evidence(tmp_path / "out")
    with pytest.raises(ServerError) as failure:
        worker.performance(URL, 1, evidence, smoke=True, after_measure=after)
    assert "secondary observer failed" in " ".join(failure.value.__notes__)
    row = json.loads((evidence.directory / "events.jsonl").read_text())
    assert len(row["completed_samples"]) == 1
    assert [value["status"] for value in row["wire_responses"]] == [200, 200, 200, 500]
    assert row["primary_error"]["type"] == "ServerError"
    assert row["finalization_errors"] == [{"type": "BrokenPipeError", "message": "secondary observer failed"}]


def test_before_measure_failure_aborts_waiting_threads_and_preserves_warmups(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    install_transport(monkeypatch, success)

    def before() -> None:
        raise OSError("boundary failed before timer")

    evidence = Evidence(tmp_path / "out")
    with pytest.raises(OSError, match="boundary failed"):
        worker.performance(URL, 4, evidence, smoke=True, before_measure=before)
    row = json.loads((evidence.directory / "events.jsonl").read_text())
    assert len(row["warmups"]) == 2
    assert row["completed_samples"] == []
    assert len(row["wire_responses"]) == 2
