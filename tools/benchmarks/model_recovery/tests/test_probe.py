"""Failure-path checks for recovery evidence and owned server shutdown."""

import argparse
import asyncio
import json
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import psutil
import pytest

from tools.benchmarks.model_recovery import probe
from tools.benchmarks.model_recovery.smoke import install_smoke


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("failure_stage", "verified_alive"),
    [("enumerate", False), ("identity", False), ("liveness", False), ("liveness", True)],
)
async def test_process_inspection_failure_does_not_skip_shutdown_or_summary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure_stage: str, verified_alive: bool
) -> None:
    monkeypatch.setattr(probe.os, "environ", probe.os.environ.copy())
    registry = SimpleNamespace(_loaded={}, loaded_model_names=[])
    app = SimpleNamespace(state=SimpleNamespace(registry=registry))
    monkeypatch.setattr(probe.AppFactory, "create_app", lambda _config: app)
    monkeypatch.setattr(probe, "snapshot", lambda _registry: {"models": {}})

    class Server:
        started = True
        should_exit = False
        shutdown_complete = False

        async def serve(self, **_kwargs: object) -> None:
            await asyncio.sleep(0)
            assert self.should_exit
            self.shutdown_complete = True

    server = Server()
    monkeypatch.setattr(probe.uvicorn, "Server", lambda _config: server)
    inspected_pids: set[int] = set()

    class Child:
        def __init__(self, pid: int) -> None:
            self.pid = pid

        def create_time(self) -> float:
            if self.pid == 12345 and (
                failure_stage == "identity" or (failure_stage == "liveness" and server.shutdown_complete)
            ):
                raise psutil.AccessDenied(self.pid)
            return float(self.pid)

        def is_running(self) -> bool:
            return verified_alive and self.pid == 12346

        def status(self) -> str:
            return psutil.STATUS_RUNNING

    class Parent:
        def children(self, **_kwargs: object) -> list[Child]:
            if failure_stage == "enumerate":
                raise psutil.AccessDenied
            return [Child(12345), Child(12346)]

    def process(pid: int) -> Parent | Child:
        if pid == probe.os.getpid():
            return Parent()
        inspected_pids.add(pid)
        return Child(pid)

    monkeypatch.setattr(probe.psutil, "Process", process)
    args = argparse.Namespace(smoke=True, deadline_s=10, output=tmp_path / "evidence")
    evidence = probe.Evidence(args.output)

    # The empty registry forces an ordinary protocol failure before warmup.
    # Process inspection must not replace that failure or bypass teardown.
    assert await probe.run(args, evidence) is False
    assert server.should_exit
    assert server.shutdown_complete
    assert any(row["event"] == "protocol_failed" for row in evidence.rows)
    inspection = next(row for row in evidence.rows if row["event"] == "before_shutdown")
    if failure_stage == "liveness":
        assert inspection["inspection_errors"] == []
        assert inspected_pids == {12345, 12346}
    else:
        assert inspection["inspection_errors"][0]["stage"] == failure_stage
    cleanup = next(row for row in evidence.rows if row["event"] == "cleanup")
    assert cleanup["okay"] is False
    assert cleanup["owned_children_alive"] == ([12346] if verified_alive else [])
    assert cleanup["inspection_errors"][0]["stage"] == failure_stage
    assert cleanup["inspection_errors"][0]["error_type"] == "AccessDenied"
    persisted = [json.loads(line) for line in (args.output / "events.jsonl").read_text().splitlines()]
    assert next(row for row in persisted if row["event"] == "before_shutdown") == inspection
    assert next(row for row in persisted if row["event"] == "cleanup") == cleanup
    summary = json.loads((args.output / "summary.json").read_text())
    assert summary["passed"] is False
    assert summary["cleanup_ok"] is False


@pytest.mark.asyncio
async def test_final_snapshot_failure_cannot_pass_a_successful_protocol(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(probe.os, "environ", probe.os.environ.copy())
    args = argparse.Namespace(
        smoke=True,
        deadline_s=30,
        output=tmp_path / "evidence",
        phase_requests=1,
        during_requests=1,
        interval_s=0,
        recovery_timeout_s=20,
    )
    evidence = probe.Evidence(args.output)
    original_snapshot = probe.snapshot

    def snapshot(registry: Any) -> dict[str, Any]:
        state = original_snapshot(registry)
        if evidence.rows[-1]["event"] == "before_shutdown":
            assert state["resident_names"] == []
            raise RuntimeError("final snapshot failed after shutdown")
        return state

    monkeypatch.setattr(probe, "snapshot", snapshot)
    with ExitStack() as stack:
        install_smoke(stack)
        assert await probe.run(args, evidence) is False

    summary = json.loads((args.output / "summary.json").read_text())
    assert summary["protocol_passed"] is True
    assert summary["crash_count"] == summary["successful_reloads"] == 1
    assert summary["passed"] is False
    assert summary["cleanup_ok"] is False
    assert all(phase["errors"] == 0 for phase in summary["healthy_requests"].values())
    failed = next(row for row in evidence.rows if row["event"] == "cleanup_failed")
    assert failed["error_type"] == "RuntimeError"
    assert failed["message"] == "final snapshot failed after shutdown"
    assert failed["owned_children_alive"] == []
    assert failed["inspection_errors"] == []
    persisted = [json.loads(line) for line in (args.output / "events.jsonl").read_text().splitlines()]
    assert next(row for row in persisted if row["event"] == "cleanup_failed") == failed
