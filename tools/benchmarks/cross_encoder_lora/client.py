"""SDK-only request capture and load loop, shared by local and isolated drivers."""

from __future__ import annotations

import math
import statistics
import threading
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from http import HTTPStatus
from typing import Any

import httpx
from sie_sdk import SIEClient
from sie_sdk._msgpack import unpackb
from sie_sdk.client.errors import ServerError

from tools.benchmarks.cross_encoder_lora.evidence import Evidence
from tools.benchmarks.cross_encoder_lora.protocol import (
    DOCUMENTS,
    MODEL,
    PERFORMANCE_FUTURE_TIMEOUT_S,
    QUERY,
    SEQUENTIAL,
    STEADY_SECONDS,
    WARMUP,
    percentile,
    require,
)

MAX_LOADING_RETRIES = 10
MAX_LOADING_DELAY = 5
REQUEST_ITEMS = len(DOCUMENTS)
RESPONSE_CAPTURE_MODE = "success-status-header-snapshots"


class Client:
    def __init__(self, url: str) -> None:
        self.responses: list[dict[str, Any] | httpx.Response] = []
        http = httpx.Client(base_url=url, timeout=30, event_hooks={"response": [self.observe]})
        self._http = http
        self.sdk = SIEClient(url, timeout_s=30, http_client=http)

    def observe(self, response: httpx.Response) -> None:
        # Keep successful wire evidence without retaining its Response/Request/
        # stream graph for the whole phase. Failure bodies are decoded later.
        self.responses.append(describe_response(response) if response.status_code == HTTPStatus.OK else response)

    def score(self, selection: str | None, label: str, *, cold: bool = False, timed: bool = False) -> dict[str, Any]:
        started = time.perf_counter_ns()
        wall_started = time.time_ns()
        first = len(self.responses)
        # score() currently does not retry LORA_LOADING. These are explicitly
        # fresh SDK calls for this pre-execution status, never transport retries.
        for attempt in range(MAX_LOADING_RETRIES + 1):
            try:
                result = self.sdk.score(
                    MODEL,
                    {"text": QUERY["text"]},
                    [{"id": f"{label}:{index}", "text": text} for index, text in enumerate(DOCUMENTS)],
                    options={"lora": selection},
                    wait_for_capacity=False,
                    max_oom_retries=0,
                    provision_timeout_s=30,
                )
                break
            except ServerError as error:
                if (
                    not cold
                    or error.status_code != HTTPStatus.SERVICE_UNAVAILABLE
                    or error.code != "LORA_LOADING"
                    or attempt == MAX_LOADING_RETRIES
                ):
                    raise
                delay = error.retry_after if error.retry_after is not None else 0.2
                require(0 <= delay <= MAX_LOADING_DELAY, "unexpected loading retry delay")
                time.sleep(delay)
        elapsed = (time.perf_counter_ns() - started) / 1e6
        row = {
            "label": label,
            "selection": selection,
            "latency_ms": elapsed,
            "start_ns": wall_started,
            "end_ns": time.time_ns(),
            "result": result,
            "responses": self.responses[first:],
        }
        if not timed:
            validate_response(row, allow_loading=cold)
        return row

    def close(self) -> None:
        try:
            self.sdk.close()
        finally:
            # Callers serialize their evidence before closing. Preserve unrelated
            # hooks, and break only the observer/history this benchmark owns.
            hooks = self._http.event_hooks["response"]
            hooks[:] = [
                hook
                for hook in hooks
                if not (
                    getattr(hook, "__self__", None) is self and getattr(hook, "__func__", None) is type(self).observe
                )
            ]
            self.responses.clear()


def describe_response(response: httpx.Response | dict[str, Any]) -> dict[str, Any]:
    if not isinstance(response, httpx.Response):
        return response
    row: dict[str, Any] = {
        "status": response.status_code,
        "timing_headers": {
            name: response.headers[name]
            for name in (
                "x-queue-time",
                "x-tokenization-time",
                "x-inference-time",
                "x-postprocessing-time",
                "x-total-time",
            )
            if name in response.headers
        },
    }
    if response.status_code >= HTTPStatus.BAD_REQUEST:
        row["content_type"] = response.headers.get("content-type")
        row["retry_after"] = response.headers.get("retry-after")
        try:
            response.read()
        except (httpx.StreamError, httpx.HTTPError) as error:
            # A partially read error stream can already have failed in the SDK.
            # Preserve that original failure and its wire metadata; diagnostic
            # re-reading must not replace it or discard earlier attempts.
            row.update(body_unavailable=True, body_read_error=type(error).__name__, code=None)
            return row
        row["body_hex"] = response.content.hex()
        try:
            body = (
                unpackb(response.content, numeric_arrays=False)
                if "msgpack" in response.headers.get("content-type", "")
                else response.json()
            )
        except ValueError as error:
            row["body_decode_error"] = type(error).__name__
            body = None
        error = body.get("error", body.get("detail", body)) if isinstance(body, dict) else None
        row["code"] = error.get("code") if isinstance(error, dict) else None
    return row


def serialize_responses(row: dict[str, Any]) -> None:
    row["responses"] = [
        describe_response(value) if isinstance(value, httpx.Response) else value for value in row["responses"]
    ]


def validate_response(row: dict[str, Any], *, allow_loading: bool = False) -> None:
    serialize_responses(row)
    statuses = row["responses"]
    require(statuses and statuses[-1]["status"] == HTTPStatus.OK, "missing successful HTTP response")
    if allow_loading:
        require(
            all(
                value["status"] == HTTPStatus.SERVICE_UNAVAILABLE and value.get("code") == "LORA_LOADING"
                for value in statuses[:-1]
            ),
            "unexpected retry during cold adapter request",
        )
    else:
        require(len(statuses) == 1, "hidden SDK retry disqualifies this request")
    scores = row["result"]["scores"]
    expected_ids = [f"{row['label']}:{i}" for i in range(REQUEST_ITEMS)]
    by_id = {item["item_id"]: item["score"] for item in scores}
    require(len(scores) == REQUEST_ITEMS and set(by_id) == set(expected_ids), "score result lost or changed item IDs")
    require([item["rank"] for item in scores] == list(range(REQUEST_ITEMS)), "score ranks changed")
    values = [by_id[key] for key in expected_ids]
    row["scores"] = values
    require(all(math.isfinite(value) for value in values), "nonfinite SDK score")
    require([item["score"] for item in scores] == sorted(values, reverse=True), "score ranks are unordered")


def performance(
    url: str,
    concurrency: int,
    evidence: Evidence,
    *,
    smoke: bool,
    warmup: int = WARMUP,
    sequential: int = SEQUENTIAL,
    steady_seconds: float = STEADY_SECONDS,
    future_timeout: float = PERFORMANCE_FUTURE_TIMEOUT_S,
    before_measure: Callable[[], None] | None = None,
    after_measure: Callable[[dict[str, float]], None] | None = None,
) -> dict[str, Any]:
    clients: list[Client] = []
    samples: list[dict[str, Any]] = []
    warmups: list[dict[str, Any]] = []
    lock = threading.Lock()
    clock: dict[str, float] = {}
    measurement_notified = False
    primary_error: BaseException | None = None

    def notify_measurement_end() -> None:
        nonlocal measurement_notified
        if "start" in clock and not measurement_notified:
            clock.setdefault("end", time.perf_counter())
            measurement_notified = True
            if after_measure is not None:
                after_measure(dict(clock))

    try:
        for _ in range(concurrency):
            clients.append(Client(url))
        for i in range(2 if smoke else warmup):
            warmups.append(clients[0].score(None, f"warm-C{concurrency}-{i}", timed=True))
        barrier = threading.Barrier(concurrency + 1)

        def lane(index: int) -> None:
            barrier.wait(timeout=30)
            counter = 0
            while (
                counter < (4 if smoke else sequential) if concurrency == 1 else time.perf_counter() < clock["deadline"]
            ):
                row = clients[index].score(None, f"C{concurrency}-lane{index}-{counter}", timed=True)
                with lock:
                    samples.append(row)
                counter += 1

        with ThreadPoolExecutor(max_workers=concurrency) as executor:
            futures = [executor.submit(lane, i) for i in range(concurrency)]
            try:
                if before_measure is not None:
                    before_measure()
            except BaseException:
                barrier.abort()
                raise
            clock["start"] = time.perf_counter()
            clock["deadline"] = clock["start"] + (0.15 if smoke else steady_seconds)
            barrier.wait(timeout=30)
            for future in futures:
                future.result(timeout=future_timeout)
        clock["end"] = time.perf_counter()
        elapsed = clock["end"] - clock["start"]
        notify_measurement_end()
        for row in [*warmups, *samples]:
            validate_response(row)
        latencies = [row["latency_ms"] for row in samples]
        return {
            "concurrency": concurrency,
            "count": len(samples),
            "elapsed_s": elapsed,
            "mean_ms": statistics.mean(latencies),
            "p95_ms": percentile(latencies, 0.95),
            "rps": len(samples) / elapsed,
            "samples": samples,
            "measurement_clock": dict(clock),
        }
    except BaseException as error:
        primary_error = error
        raise
    finally:
        # Finalize each owned responsibility independently. A broken telemetry
        # handshake must not replace an SDK failure or erase its wire attempts.
        finalization_errors: list[BaseException] = []

        def finalize(action: Callable[[], None]) -> None:
            try:
                action()
            except BaseException as error:  # noqa: BLE001 -- finalize all owned resources, then re-raise
                finalization_errors.append(error)

        finalize(notify_measurement_end)
        for row in [*warmups, *samples]:
            finalize(lambda row=row: serialize_responses(row))
        finalize(
            lambda: evidence.emit(
                "performance_attempt",
                concurrency=concurrency,
                completed_samples=samples,
                warmups=warmups,
                wire_responses=[describe_response(response) for client in clients for response in client.responses],
                primary_error=None
                if primary_error is None
                else {"type": type(primary_error).__name__, "message": str(primary_error)},
                finalization_errors=[
                    {"type": type(error).__name__, "message": str(error)} for error in finalization_errors
                ],
            )
        )
        for client in clients:
            finalize(client.close)
        if finalization_errors:
            if primary_error is not None:
                for error in finalization_errors:
                    primary_error.add_note(f"Secondary benchmark finalization failure: {type(error).__name__}: {error}")
            else:
                raise finalization_errors[0]
