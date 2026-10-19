"""Admission reserves the complete worker wait and both cleanup grace periods."""

from __future__ import annotations

import argparse
import json
import signal
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from tools.benchmarks.cross_encoder_lora import runner


def arguments(tmp_path: Path, budget: float, trial_limit: float = 120) -> argparse.Namespace:
    return argparse.Namespace(
        output=tmp_path / "evidence",
        inputs=tmp_path / "inputs",
        fixtures=tmp_path / "fixtures",
        snapshot=None,
        smoke=True,
        environment_receipt=None,
        deadline_s=budget,
        trial_deadline_s=trial_limit,
    )


def setup_sources(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(runner, "extract", lambda *_: {"sources": {"candidate": {"commit": "frozen"}}})
    monkeypatch.setattr(runner, "harness_manifest", dict)


@pytest.mark.parametrize("trial_limit", [120, 270])
@pytest.mark.parametrize(("margin", "admitted"), [(-0.001, False), (0.0, False), (0.001, True)])
def test_trial_admission_reserves_full_cleanup(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    trial_limit: float,
    margin: float,
    admitted: bool,
) -> None:
    setup_sources(monkeypatch)
    monkeypatch.setattr(runner.time, "monotonic", lambda: 100.0)
    waits = []
    started = []

    class CompletedProcess:
        returncode = 0

        def wait(self, *, timeout: float) -> int:
            waits.append(timeout)
            return self.returncode

        def poll(self) -> int:
            return self.returncode

    def start(command: list[str], **_kwargs: Any) -> CompletedProcess:
        started.append(command)
        output = Path(command[command.index("--output") + 1])
        output.mkdir()
        (output / "trial.json").write_text(
            json.dumps({"pair": 0, "arm": "R", "passed": True, "shutdown_complete": True})
        )
        return CompletedProcess()

    monkeypatch.setattr(runner.subprocess, "Popen", start)
    args = arguments(tmp_path, trial_limit + 90 + margin, trial_limit)
    if admitted:
        assert runner.run(args)["passed"]
        assert waits == [trial_limit + 45]
    else:
        with pytest.raises(AssertionError, match="insufficient time"):
            runner.run(args)
        terminal = json.loads((args.output / "summary.json").read_text())
        assert not terminal["passed"]
        assert terminal["trials"] == []
    assert bool(started) is admitted
    terminal = json.loads((args.output / "summary.json").read_text())
    assert terminal["runner_clock"]["monotonic_start_s"] == 100
    assert terminal["runner_clock"]["budget_s"] == args.deadline_s
    assert terminal["runner_clock"]["deadline_monotonic_s"] == 100 + args.deadline_s
    assert terminal["runner_clock"]["wall_start_ns"] > 0
    assert len(terminal["admissions"]) == 1
    admission = terminal["admissions"][0]
    assert (admission["pair"], admission["arm"], admission["monotonic_s"]) == (0, "R", 100)
    assert admission["trial_limit_s"] == trial_limit
    assert admission["reserve_s"] == trial_limit + 90
    assert admission["margin_s"] == pytest.approx(margin, abs=1e-9)
    assert admission["admitted"] is admitted


@pytest.mark.parametrize("trial_limit", [120, 270])
def test_full_timeout_cleanup_finishes_before_budget_and_retains_partial_trial(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    trial_limit: float,
) -> None:
    setup_sources(monkeypatch)
    clock = [100.0]
    monkeypatch.setattr(runner.time, "monotonic", lambda: clock[0])
    actions = []

    class HungProcess:
        returncode = None

        def wait(self, *, timeout: float) -> int:
            actions.append(("wait", timeout))
            clock[0] += timeout
            if len([action for action, _ in actions if action == "wait"]) < 3:
                raise subprocess.TimeoutExpired("owned-worker", timeout)
            self.returncode = -9
            return self.returncode

        def poll(self) -> None:
            return None

        def send_signal(self, signum: int) -> None:
            actions.append(("signal", signum))

        def kill(self) -> None:
            actions.append(("kill", None))

    def start(command: list[str], **_kwargs: Any) -> HungProcess:
        output = Path(command[command.index("--output") + 1])
        output.mkdir()
        (output / "trial.json").write_text(
            json.dumps(
                {
                    "pair": 0,
                    "arm": "R",
                    "passed": False,
                    "shutdown_complete": False,
                    "performance": {"1": {"count": 7}},
                }
            )
        )
        return HungProcess()

    monkeypatch.setattr(runner.subprocess, "Popen", start)
    args = arguments(tmp_path, trial_limit + 91, trial_limit)
    with pytest.raises(subprocess.TimeoutExpired):
        runner.run(args)
    assert actions == [
        ("wait", trial_limit + 45),
        ("signal", signal.SIGTERM),
        ("wait", 35),
        ("kill", None),
        ("wait", 10),
    ]
    assert clock[0] == 100 + trial_limit + 90
    assert clock[0] < 100 + args.deadline_s
    terminal = json.loads((args.output / "summary.json").read_text())
    assert not terminal["passed"]
    assert terminal["admissions"][0]["margin_s"] == 1
    assert terminal["admissions"][0]["admitted"]
    trial = terminal["trials"][0]
    assert trial["performance"]["1"]["count"] == 7
    assert trial["process_returncode"] == -9
    assert trial["runner_error"]["type"] == "TimeoutExpired"


def test_default_cli_budget_matches_longer_qualification(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "argv", ["runner", "--inputs", "i", "--fixtures", "f", "--output", "o"])
    monkeypatch.setattr(runner.signal, "signal", lambda *_: None)
    captured = []
    monkeypatch.setattr(runner, "run", captured.append)
    runner.main()
    assert len(captured) == 1
    assert (captured[0].deadline_s, captured[0].trial_deadline_s) == (4200, 270)


@pytest.mark.parametrize(("flag", "value"), [("--deadline-s", "4200.001"), ("--trial-deadline-s", "270.001")])
def test_cli_rejects_limits_above_frozen_bounds(monkeypatch: pytest.MonkeyPatch, flag: str, value: str) -> None:
    monkeypatch.setattr(sys, "argv", ["runner", "--inputs", "i", "--fixtures", "f", "--output", "o", flag, value])
    started = []
    monkeypatch.setattr(runner, "run", started.append)
    with pytest.raises(AssertionError, match="deadline outside bounded protocol"):
        runner.main()
    assert not started
