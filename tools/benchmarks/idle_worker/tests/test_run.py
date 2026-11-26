"""The timing probe follows the SDK's real extraction transport path."""

import argparse
import json
from pathlib import Path

import httpx
import pytest
from sie_sdk import SIEClient
from sie_sdk._msgpack import packb
from sie_sdk.types import Item

from tools.benchmarks.idle_worker import run
from tools.benchmarks.idle_worker.run import distribution, timing_sample


def test_timing_hook_follows_sdk_extract_route() -> None:
    samples = []

    def response_for(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            request=request,
            headers={
                "X-Queue-Time": "3.81",
                "X-Inference-Time": "20.0",
                "X-Total-Time": "24.0",
                "Content-Type": "application/msgpack",
            },
            content=packb(
                {
                    "model": "knowledgator/gliclass-large-v1.0",
                    "items": [{"id": "probe", "entities": [], "classifications": [{"label": "billing", "score": 0.9}]}],
                }
            ),
        )

    transport = httpx.Client(
        base_url="http://127.0.0.1:8373",
        transport=httpx.MockTransport(response_for),
        event_hooks={"response": [lambda response: samples.append(timing_sample(response))]},
    )
    with SIEClient("http://127.0.0.1:8373", http_client=transport) as client:
        client.extract("knowledgator/gliclass-large-v1.0", Item(text="Refund please", id="probe"), labels=["billing"])
    assert samples == [{"queue_ms": 3.81, "inference_ms": 20.0, "total_ms": 24.0}]


@pytest.mark.parametrize(("count", "p95"), [(1, 1), (20, 19), (100, 95), (101, 96), (300, 285)])
def test_distribution_uses_nearest_rank_p95(count: int, p95: int) -> None:
    assert distribution(list(reversed(range(1, count + 1))))["p95"] == p95


@pytest.mark.parametrize("failure_type", [RuntimeError, KeyboardInterrupt])
@pytest.mark.parametrize("persistence_fails", [False, True])
def test_failed_trial_preserves_completed_phases_and_stops_server(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure_type: type[BaseException], persistence_fails: bool
) -> None:
    stopped = []

    class Process:
        returncode = None

        def terminate(self) -> None:
            stopped.append("terminate")

        def wait(self, timeout: int) -> None:
            stopped.append(timeout)

    monkeypatch.setattr(run.subprocess, "Popen", lambda *args, **kwargs: Process())
    monkeypatch.setattr(run, "wait_for_server", lambda *args: None)
    real_client = httpx.Client

    def response_for(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            request=request,
            headers={
                "X-Queue-Time": "0.12",
                "X-Inference-Time": "20",
                "X-Total-Time": "21",
                "Content-Type": "application/msgpack",
            },
            content=packb(
                {
                    "model": "probe",
                    "items": [
                        {"id": "sequential:0", "entities": [], "classifications": [{"label": "billing", "score": 0.9}]}
                    ],
                }
            ),
        )

    monkeypatch.setattr(
        run.httpx,
        "Client",
        lambda *args, **kwargs: real_client(*args, **kwargs, transport=httpx.MockTransport(response_for)),
    )
    failure = failure_type("burst failed after sequential completed")
    trace = tmp_path / "00-candidate-batches.jsonl"
    trace_record = '{"request_ids": ["sequential:0"], "size": 1}\n'

    def measure(client: SIEClient, model: str, phase: str, concurrency: int, **kwargs) -> dict:
        if phase.startswith("warmup"):
            return {}
        if phase == "sequential":
            client.extract(model, Item(text="Refund please", id="sequential:0"), labels=["billing"])
            trace.write_text(trace_record)
            return {"phase": phase, "requests": 1}
        raise failure

    monkeypatch.setattr(run, "measure", measure)
    real_write_text = Path.write_text

    def write_text(path: Path, data: str, *args, **kwargs) -> int:
        if persistence_fails and '"status": "failed"' in data:
            real_write_text(path, "incomplete")
            raise OSError("disk unavailable while recording failure")
        return real_write_text(path, data, *args, **kwargs)

    monkeypatch.setattr(Path, "write_text", write_text)
    args = argparse.Namespace(model="probe", port=8373, device="cpu", sequential=1, duration=0.01)
    with pytest.raises(failure_type) as caught:
        run.trial(args, "candidate", 0, tmp_path, tmp_path)

    assert caught.value is failure
    assert stopped == ["terminate", 40]
    result = json.loads((tmp_path / "00-candidate.json").read_text())
    assert result["status"] == ("running" if persistence_fails else "failed")
    assert result["phases"][0]["phase"] == "sequential"
    assert result["phases"][0]["samples"] == [{"queue_ms": 0.12, "inference_ms": 20.0, "total_ms": 21.0}]
    assert trace.read_text() == trace_record
    if persistence_fails:
        assert any("disk unavailable" in note for note in caught.value.__notes__)
