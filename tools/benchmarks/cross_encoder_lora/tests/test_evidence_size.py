"""Raw observations remain revalidatable without multiplying diagnostic copies."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from tools.benchmarks.cross_encoder_lora import runner
from tools.benchmarks.cross_encoder_lora.evidence import atomic_json, digest, vector_delta
from tools.benchmarks.cross_encoder_lora.protocol import ORDERS, RESTORE_ATOL
from tools.benchmarks.cross_encoder_lora.tests.test_runner_budget import arguments, setup_sources
from tools.benchmarks.cross_encoder_lora.worker import ORDER, matching_vector


def corpus() -> tuple[list[float], dict[str, Any]]:
    scores = [1.0, 2.0, 3.0, 4.0]
    oracles = {
        f"workload-{copies:02d}": {
            "copies": copies,
            "scores": [scores[index] for index in ORDER for _ in range(copies)],
        }
        for copies in range(1, 17)
    }
    return scores, oracles


def oracle_vector(oracles: dict[str, Any], reference: dict[str, Any]) -> list[float]:
    # Deliberately reconstruct independently of the matcher, in request order.
    oracle = oracles[reference["workload_sha256"]]
    assert reference["copies"] == oracle["copies"]
    offset = reference["request_position"]
    groups = {
        item: oracle["scores"][index * oracle["copies"] : (index + 1) * oracle["copies"]]
        for index, item in enumerate(ORDER)
    }
    return [groups[item][offset] for item in range(4)]


def test_all_136_matches_have_bounded_reconstructible_witness() -> None:
    scores, oracles = corpus()
    reference = matching_vector(scores, oracles)
    assert reference == {"workload_sha256": "workload-01", "copies": 1, "request_position": 0}
    assert matching_vector(scores, dict(reversed(list(oracles.items())))) == reference
    raw = [
        {
            "label": str(index),
            "scores": scores,
            "latency_ms": 1.0 + index / 1000,
            "responses": [{"status": 200}],
            "diagnostic_matching_oracle": matching_vector(scores, oracles),
        }
        for index in range(1000)
    ]
    encoded = json.dumps({"shape_oracles": oracles, "samples": raw})
    assert len(encoded) < 260_000  # Independent of repeating all 136 witnesses per response.
    recovered = json.loads(encoded)
    for original, row in zip(raw, recovered["samples"], strict=True):
        assert row == original
        witness = oracle_vector(recovered["shape_oracles"], row["diagnostic_matching_oracle"])
        assert vector_delta(row["scores"], witness) <= RESTORE_ATOL
    all_matches = [
        {"workload_sha256": key, "copies": oracle["copies"], "request_position": position}
        for key, oracle in recovered["shape_oracles"].items()
        for position in range(oracle["copies"])
        if vector_delta(
            scores,
            oracle_vector(
                recovered["shape_oracles"],
                {"workload_sha256": key, "copies": oracle["copies"], "request_position": position},
            ),
        )
        <= RESTORE_ATOL
    ]
    assert len(all_matches) == 136  # Ambiguity is reconstructible, not asserted away.
    assert matching_vector([1.0, 2.0, 3.0, 4.0 + 2 * RESTORE_ATOL], oracles) is None


def terminal(directory: Path, pair: int, arm: str, *, failed: bool = False) -> dict[str, Any]:
    scores, oracles = corpus()
    rows = [
        {
            "label": str(index),
            "scores": scores,
            "latency_ms": 10.0,
            "responses": [
                {"status": 503, "code": "TEST_FAILURE", "body": "retained error"} if failed else {"status": 200}
            ],
        }
        for index in range(500)
    ]
    trial = {
        "pair": pair,
        "arm": arm,
        "passed": not failed,
        "shutdown_complete": not failed,
        "shape_oracles": oracles,
        "performance": {
            str(concurrency): {
                "count": len(rows),
                "mean_ms": 10.0,
                "p95_ms": 10.0,
                "rps": 100.0,
                "samples": rows,
                "future_raw_field": rows,
            }
            for concurrency in (1, 4, 16)
        },
    }
    if failed:
        trial["error"] = {"type": "RuntimeError", "message": "retained failure"}
    directory.mkdir(parents=True)
    atomic_json(directory / "trial.json", trial)
    return trial


def test_summary_size_is_bounded_and_final_checks_reopen_every_trial(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    setup_sources(monkeypatch)
    monkeypatch.setattr(runner, "extract", lambda *_: {"sources": {arm: {"commit": arm} for arm in ("B", "candidate")}})
    monkeypatch.setattr(runner.time, "monotonic", lambda: 100.0)

    class CompletedProcess:
        returncode = 0

        def wait(self, *, timeout: float) -> int:
            return self.returncode

        def poll(self) -> int:
            return self.returncode

    def start(command: list[str], **_kwargs: Any) -> CompletedProcess:
        def value(flag: str) -> str:
            return command[command.index(flag) + 1]

        terminal(Path(value("--output")), int(value("--pair")), value("--arm"))
        return CompletedProcess()

    full_records = []

    def inspect(trials: list[dict[str, Any]]) -> dict[str, Any]:
        full_records.extend(trials)
        assert len(trials) == 18
        for trial in trials:
            assert len(trial["shape_oracles"]) == 16
            for phase in trial["performance"].values():
                assert len(phase["samples"]) == 500
                assert phase["samples"][0]["scores"] == [1.0, 2.0, 3.0, 4.0]
                assert phase["samples"][0]["responses"] == [{"status": 200}]
        return {"passed": True}

    monkeypatch.setattr(runner.subprocess, "Popen", start)
    monkeypatch.setattr(runner, "cross_checks", inspect)
    args = arguments(tmp_path, 3000)
    args.smoke = False
    summary = runner.run(args)
    assert summary["schema"] == 2
    assert summary["passed"]
    assert len(full_records) == 18
    encoded = (args.output / "summary.json").read_bytes()
    assert len(encoded) < 30_000
    assert b'"samples"' not in encoded
    assert b'"shape_oracles"' not in encoded
    assert b'"future_raw_field"' not in encoded
    for record, (pair, arm) in zip(
        summary["trials"], ((i, arm) for i, order in enumerate(ORDERS) for arm in order), strict=True
    ):
        assert record["terminal_record"] == f"trial-{pair}-{arm}/trial.json"
        assert record["terminal_record_sha256"] == digest(args.output / record["terminal_record"])


def test_failed_raw_record_survives_compaction_and_refs_reject_tampering(tmp_path: Path) -> None:
    path = tmp_path / "trial-0-R/trial.json"
    trial = terminal(path.parent, 0, "R", failed=True)
    trial.update(
        terminal_record="trial-0-R/trial.json",
        terminal_record_sha256=digest(path),
        process_returncode=-9,
        runner_error={"type": "TimeoutExpired", "message": "owned worker stopped"},
    )
    record = runner.trial_summary(trial)
    assert not record["passed"]
    assert not record["shutdown_complete"]
    assert record["error"] == trial["error"]
    assert record["runner_error"] == trial["runner_error"]
    assert record["process_returncode"] == -9
    raw = runner.load_trial(tmp_path, record)
    assert raw["performance"]["1"]["samples"] == trial["performance"]["1"]["samples"]
    changed = json.loads(json.dumps(record))
    changed["performance"]["1"]["mean_ms"] = 9.0
    with pytest.raises(AssertionError, match="summary differs"):
        runner.load_trial(tmp_path, changed)
    with pytest.raises(AssertionError, match="record path"):
        runner.load_trial(tmp_path, {**record, "terminal_record": "../trial.json"})
    path.write_text(path.read_text() + "\n")
    with pytest.raises(AssertionError, match="record hash"):
        runner.load_trial(tmp_path, record)
    raw["pair"] = 1
    atomic_json(path, raw)
    with pytest.raises(AssertionError, match="identity"):
        runner.load_trial(tmp_path, {**record, "terminal_record_sha256": digest(path)})


def test_runner_failure_override_never_turns_into_success(tmp_path: Path) -> None:
    path = tmp_path / "trial-0-R/trial.json"
    trial = terminal(path.parent, 0, "R")
    trial.update(
        terminal_record="trial-0-R/trial.json",
        terminal_record_sha256=digest(path),
        passed=False,
        runner_error={"type": "TimeoutExpired", "message": "owned worker stopped"},
    )
    record = runner.trial_summary(trial)
    reopened = runner.load_trial(tmp_path, record)
    assert not reopened["passed"]
    assert reopened["runner_error"] == record["runner_error"]
    assert json.loads(path.read_text())["passed"]  # The original raw file remains unchanged.
    assert not record["passed"]
    with pytest.raises(AssertionError, match="lost its failed verdict"):
        runner.load_trial(tmp_path, {**record, "passed": True})
