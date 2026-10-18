"""Actual archived loader/SDK triplets and fail-closed group semantics."""

# All subprocesses use the current interpreter and owned temporary artifacts.
# ruff: noqa: S603

from __future__ import annotations

import argparse
import asyncio
import copy
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import cast

import pytest

from tools.benchmarks.cross_encoder_lora import interleaved, prepare
from tools.benchmarks.cross_encoder_lora.evidence import atomic_json, digest
from tools.benchmarks.cross_encoder_lora.fixtures import prepare as fixtures_prepare
from tools.benchmarks.cross_encoder_lora.group_control import envelope, selection, validate_binding, validate_source
from tools.benchmarks.cross_encoder_lora.protocol import BASE_REVISION
from tools.benchmarks.cross_encoder_lora.runner import ROOT, source_environment
from tools.benchmarks.cross_encoder_lora.tests.test_probe import tiny_snapshot


def require_baseline_history() -> None:
    available = subprocess.run(["git", "cat-file", "-e", BASE_REVISION], cwd=ROOT, capture_output=True, check=False)  # noqa: S607
    if available.returncode:
        pytest.skip("immutable archive smoke requires the documented baseline commit in git history")


def test_strict_group_admission() -> None:
    assert not interleaved.admission(100, 789)["admitted"]
    assert not interleaved.admission(100, 790)["admitted"]
    assert interleaved.admission(100, 791)["admitted"]


@pytest.mark.parametrize("role", ["B", "N", "R"])
def test_explicit_role_configuration_source(role: str) -> None:
    assert selection("baseline-pilot", role) == ("B", "B")
    assert selection("qualification", role) == (("B", "B") if role == "B" else (role, "candidate"))


def binding(tmp_path: Path) -> dict:
    start = time.monotonic()
    return {
        "schema": 1,
        "schedule": "whole-phase",
        "mode": "baseline-pilot",
        "pair": 0,
        "logical_role": "R",
        "actual_arm": "B",
        "source_key": "B",
        "source_commit": BASE_REVISION,
        "nonce": "a" * 32,
        "coordinator_pid": 1,
        "group_started": start,
        "work_deadline": start + 30,
        "hard_deadline": start + 120,
        "active_limit_s": 270,
        "phase_limit_s": 120,
        "output": str(tmp_path),
    }


@pytest.mark.parametrize(
    "change",
    [
        {"actual_arm": "R"},
        {"source_key": "candidate"},
        {"source_commit": "f" * 40},
        {"phase_limit_s": 121},
        {"active_limit_s": 271},
        {"pair": True},
    ],
)
def test_binding_rejects_relabeling_and_budget_changes(tmp_path: Path, change: dict) -> None:
    value = binding(tmp_path)
    validate_binding(value)
    value.update(change)
    with pytest.raises(AssertionError):
        validate_binding(value)


def test_qualification_resident_cannot_be_base(tmp_path: Path) -> None:
    value = {**binding(tmp_path), "mode": "qualification"}
    with pytest.raises(AssertionError, match="mismatch"):
        validate_binding(value)


def test_manifest_and_extracted_source_hashes_checked_before_load(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "uv.lock").write_text("fixed")
    archive = tmp_path / "B.tar"
    archive.write_bytes(b"archive bytes")
    manifest = tmp_path / "manifest.json"
    atomic_json(
        manifest,
        {
            "schema": 1,
            "dependency_lock_identical": True,
            "sources": {"B": {"commit": BASE_REVISION, "archive": "B.tar", "sha256": digest(archive)}},
        },
    )
    value = {
        **binding(tmp_path),
        "source_root": str(source),
        "source_manifest": str(manifest),
        "source_manifest_sha256": digest(manifest),
        "archive_sha256": digest(archive),
        "source_files": {"uv.lock": digest(source / "uv.lock")},
    }
    validate_source(value, source, "B", 0)
    with pytest.raises(AssertionError, match="arguments"):
        validate_source(value, source, "R", 0)
    (source / "uv.lock").write_text("changed")
    with pytest.raises(AssertionError, match="source file"):
        validate_source(value, source, "B", 0)


@pytest.mark.parametrize("mode", ["baseline-pilot", "qualification"])
def test_real_archived_whole_phase_triplet(tmp_path: Path, mode: str) -> None:
    require_baseline_history()
    snapshot = tiny_snapshot(tmp_path / "model")
    fixtures = tmp_path / "fixtures"
    fixtures_prepare(fixtures, snapshot, smoke=True)
    inputs = tmp_path / "inputs"
    prepare.prepare(ROOT, inputs, "HEAD")
    output = tmp_path / "out"
    command = [
        sys.executable,
        "-m",
        "tools.benchmarks.cross_encoder_lora.interleaved",
        "--inputs",
        str(inputs),
        "--fixtures",
        str(fixtures),
        "--output",
        str(output),
        "--mode",
        mode,
        "--smoke",
        "--group-work-s",
        "120",
        "--deadline-s",
        "240",
    ]
    result = subprocess.run(
        command, cwd=ROOT, env=source_environment(ROOT), capture_output=True, text=True, timeout=225, check=False
    )
    assert result.returncode == 0, (
        result.stdout
        + result.stderr
        + "\n"
        + "\n".join(path.read_text()[-8000:] for path in sorted(output.glob("*.log")))
    )
    summary = json.loads((output / "summary.json").read_text())
    assert summary["completed_valid"]
    assert summary["passed"]
    assert summary["qualification_passed"] is False
    group = json.loads((output / "group-0.json").read_text())
    assert [event["stage"] for event in group["leases"]] == [
        stage for stage in ("prepare", "C1", "C4", "C16", "finish") for _ in range(3)
    ]
    for left, right in zip(group["leases"], group["leases"][1:], strict=False):
        assert left["ended"] <= right["started"]
    phase_end = max(event["ended"] for event in group["leases"] if event["stage"].startswith("C"))
    assert all(event["started"] >= phase_end for event in group["leases"] if event["stage"] == "finish")
    assert len(group["cleanup"]) == 3
    assert all(row["reaped"] and row["returncode"] == 0 for row in group["cleanup"])
    trials = [json.loads((output / record["terminal_record"]).read_text()) for record in summary["trials"]]
    interleaved.validate_trials(trials, mode, smoke=True)
    assert {row["arm"] for row in trials} == {"B", "N", "R"}
    for trial in trials:
        actual, source_key = selection(mode, trial["arm"])
        assert trial["actual_arm"] == actual
        assert trial["group"]["source_key"] == source_key
        assert trial["manager_present"] == (actual == "R")
        assert trial["loadgen"]["passed"]
        assert trial["loadgen"]["reaped"]
        assert Path(trial["environment"]["adapter_source"]).is_relative_to(output / "sources" / source_key)
        assert all(
            phase["client_observation"]["gc"]["mode"] == "boundary-only" for phase in trial["performance"].values()
        )
    invalid = copy.deepcopy(trials)
    invalid[-1]["actual_arm"] = "B" if mode == "qualification" else "R"
    with pytest.raises(AssertionError, match="configuration"):
        interleaved.validate_trials(invalid, mode, smoke=True)


def test_incomplete_group_cannot_pass(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    args = argparse.Namespace(
        mode="baseline-pilot",
        deadline_s=1000,
        group_work_s=600,
        output=tmp_path / "out",
        inputs=tmp_path,
        fixtures=tmp_path,
        environment_receipt=None,
        smoke=False,
    )
    monkeypatch.setattr(interleaved, "extract", lambda *_: {})

    async def fail(*_):
        raise RuntimeError("owned child failed")

    monkeypatch.setattr(interleaved, "run_group", fail)
    with pytest.raises(RuntimeError, match="owned child"):
        asyncio.run(interleaved.run(args))
    value = json.loads((args.output / "summary.json").read_text())
    assert not value["completed_valid"]
    assert not value["qualification_passed"]
    assert not value["passed"]


@pytest.mark.asyncio
@pytest.mark.parametrize("mutation", ["lease", "ordinal", "stage", "phase"])
async def test_phase_ack_must_bind_exact_lease_and_persisted_payload(tmp_path: Path, mutation: str) -> None:
    value = binding(tmp_path)
    (tmp_path / "loadgen").mkdir()
    path = tmp_path / "loadgen/phase-C1.json"
    path.write_text("{}")
    peer = interleaved.Peer(value, (tmp_path / "log").open("wb"))
    peer.ordinal = 1
    peer.reader = asyncio.StreamReader()

    class Writer:
        def write(self, raw):
            request = json.loads(raw)
            ack = envelope(
                value,
                event="ack",
                stage="C1",
                ordinal=1,
                lease=request["lease"],
                monotonic_s=time.monotonic(),
                quiet={"pending": 0, "in_flight": 0},
                record={"concurrency": 1, "path": "phase-C1.json", "bytes": 2, "sha256": digest(path)},
            )
            if mutation == "phase":
                ack["record"]["concurrency"] = 4
            elif mutation == "ordinal":
                ack[mutation] = 0
            else:
                ack[mutation] = "wrong"
            assert peer.reader is not None
            peer.reader.feed_data((json.dumps(ack) + "\n").encode())

        async def drain(self):
            pass

    peer.writer = cast("asyncio.StreamWriter", Writer())
    try:
        with pytest.raises(AssertionError):
            await peer.lease("C1")
        assert peer.ordinal == 1
    finally:
        peer.log.close()


def test_coordinator_full_file_hashes_wait_for_all_nine_timers(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    peers = []
    read = []
    for role in "BNR":
        directory = tmp_path / role
        (directory / "loadgen").mkdir(parents=True)
        peer = interleaved.Peer({"logical_role": role, "output": str(directory)}, (directory / "log").open("wb"))
        peer.events = [{}]
        for c in (1, 4, 16):
            path = directory / "loadgen" / f"phase-C{c}.json"
            path.write_text("{}")
            peer.events.append({"ack": {"record": {"path": path.name, "bytes": 2, "sha256": digest(path)}}})
        peer.ordinal = 4
        peers.append(peer)
    original = interleaved.digest

    def observed(path):
        read.append(path)
        return original(path)

    monkeypatch.setattr(interleaved, "digest", observed)
    try:
        peers[-1].ordinal = 3
        with pytest.raises(AssertionError, match="timers incomplete"):
            interleaved.verify_phase_files(peers)
        assert not read
        peers[-1].ordinal = 4
        interleaved.verify_phase_files(peers)
        assert len(read) == 9
        peers[-1].events[-1]["ack"]["record"]["sha256"] = "f" * 64
        with pytest.raises(AssertionError, match="persistence"):
            interleaved.verify_phase_files(peers)
    finally:
        for peer in peers:
            peer.log.close()


def test_actual_coordinator_term_reaps_archived_workers_and_sdk_children(tmp_path: Path) -> None:
    require_baseline_history()
    snapshot = tiny_snapshot(tmp_path / "model")
    fixtures = tmp_path / "fixtures"
    fixtures_prepare(fixtures, snapshot, smoke=True)
    inputs = tmp_path / "inputs"
    prepare.prepare(ROOT, inputs, "HEAD")
    output = tmp_path / "out"
    command = [
        sys.executable,
        "-m",
        "tools.benchmarks.cross_encoder_lora.interleaved",
        "--inputs",
        str(inputs),
        "--fixtures",
        str(fixtures),
        "--output",
        str(output),
        "--mode",
        "baseline-pilot",
        "--smoke",
        "--group-work-s",
        "120",
        "--deadline-s",
        "240",
    ]
    with (tmp_path / "coordinator.log").open("wb") as log:
        process = subprocess.Popen(command, cwd=ROOT, env=source_environment(ROOT), stdout=log, stderr=log)
        try:
            deadline = time.monotonic() + 60
            group_path = output / "group-0.json"
            while True:
                assert process.poll() is None, (tmp_path / "coordinator.log").read_text()
                assert time.monotonic() < deadline
                if group_path.exists():
                    group = json.loads(group_path.read_text())
                    if len(group["leases"]) >= 4:
                        break
                time.sleep(0.01)
            completed = output / "trial-0-B/loadgen/phase-C1.json"
            before = digest(completed)
            process.send_signal(signal.SIGTERM)
            assert process.wait(timeout=20) != 0
            group = json.loads(group_path.read_text())
            assert not group["completed_valid"]
            assert group["error"]["type"] == "CancelledError"
            assert digest(completed) == before
            assert all(row["reaped"] for row in group["cleanup"])
            for row in group["cleanup"]:
                with pytest.raises(ProcessLookupError):
                    os.kill(row["pid"], 0)
            for role in "BNR":
                receipt = json.loads((output / f"trial-0-{role}/loadgen-process.json").read_text())
                assert receipt["reaped"]
                with pytest.raises(ProcessLookupError):
                    os.kill(receipt["identity"]["pid"], 0)
        finally:
            if process.poll() is None:
                process.kill()
                process.wait(timeout=5)
