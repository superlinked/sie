"""Six fresh resident triplets, one whole-phase lease at a time; no provisioning."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import signal
import socket
import sys
import time
import uuid
from pathlib import Path
from types import FunctionType
from typing import Any

from tools.benchmarks.cross_encoder_lora.evidence import atomic_json, digest
from tools.benchmarks.cross_encoder_lora.group_control import (
    GROUP_CLEANUP_S,
    GROUP_WORK_S,
    LIMIT,
    MODES,
    PHASE_LIMIT_S,
    RUNNER_LIMIT_S,
    STAGES,
    decode,
    envelope,
    selection,
    validate_binding,
)
from tools.benchmarks.cross_encoder_lora.protocol import MAX_TRIAL_SECONDS, ORDERS, compare, require
from tools.benchmarks.cross_encoder_lora.runner import (
    cross_checks,
    extract,
    harness_manifest,
    load_trial,
    source_environment,
    trial_summary,
)


def admission(now: float, deadline: float, work_s: float = GROUP_WORK_S) -> dict[str, Any]:
    require(0 < work_s <= GROUP_WORK_S, "group work cap differs")
    margin = deadline - (now + work_s + GROUP_CLEANUP_S)
    return {
        "monotonic_s": now,
        "deadline_monotonic_s": deadline,
        "reserve_s": work_s + GROUP_CLEANUP_S,
        "margin_s": margin,
        "admitted": margin > 0,
    }


def binding_for(
    args: argparse.Namespace,
    manifest: dict[str, Any],
    pair: int,
    role: str,
    nonce: str,
    started: float,
) -> dict[str, Any]:
    arm, source_key = selection(args.mode, role)
    source = (args.output / "sources" / source_key).resolve()
    record = manifest["sources"][source_key]
    value = {
        "schema": 1,
        "schedule": "whole-phase",
        "mode": args.mode,
        "pair": pair,
        "logical_role": role,
        "actual_arm": arm,
        "source_key": source_key,
        "source_commit": record["commit"],
        "archive_sha256": record["sha256"],
        "source_root": str(source),
        "source_manifest": str((args.inputs / "manifest.json").resolve()),
        "source_manifest_sha256": digest(args.inputs / "manifest.json"),
        "source_files": {
            str(path.relative_to(source)): digest(path)
            for path in sorted(source.rglob("*"))
            if path.is_file() and "__pycache__" not in path.parts and path.suffix != ".pyc"
        },
        "nonce": nonce,
        "coordinator_pid": os.getpid(),
        "group_started": started,
        "work_deadline": started + args.group_work_s,
        "hard_deadline": started + args.group_work_s + GROUP_CLEANUP_S,
        "active_limit_s": MAX_TRIAL_SECONDS,
        "phase_limit_s": PHASE_LIMIT_S,
        "output": str((args.output / f"trial-{pair}-{role}").resolve()),
    }
    validate_binding(value)
    return value


class Peer:
    def __init__(self, binding: dict[str, Any], log: Any) -> None:
        self.binding, self.log = binding, log
        self.process: asyncio.subprocess.Process | None = None
        self.reader: asyncio.StreamReader | None = None
        self.writer: asyncio.StreamWriter | None = None
        self.active_s = 0.0
        self.ordinal = 0
        self.events: list[dict[str, Any]] = []
        self.hello: dict[str, Any] | None = None
        self.finishing = False
        self.stopping = False

    async def start(self, args: argparse.Namespace, binding_path: Path) -> None:
        parent, child = socket.socketpair()
        try:
            command = [
                sys.executable,
                "-m",
                "tools.benchmarks.cross_encoder_lora.group_control",
                "--binding",
                str(binding_path),
                "--control-fd",
                str(child.fileno()),
                "--source",
                self.binding["source_root"],
                "--fixtures",
                str(args.fixtures.resolve()),
                "--output",
                self.binding["output"],
                "--arm",
                self.binding["actual_arm"],
                "--pair",
                str(self.binding["pair"]),
            ]
            if args.snapshot:
                command.extend(["--snapshot", str(args.snapshot.resolve())])
            if args.smoke:
                command.append("--smoke")
            self.process = await asyncio.create_subprocess_exec(
                *command,
                cwd=self.binding["source_root"],
                env=source_environment(Path(self.binding["source_root"])),
                stdout=self.log,
                stderr=asyncio.subprocess.STDOUT,
                pass_fds=(child.fileno(),),
                close_fds=True,
                start_new_session=True,
            )
            self.reader, self.writer = await asyncio.open_connection(sock=parent, limit=LIMIT)
        except BaseException:
            parent.close()
            raise
        finally:
            child.close()

    async def read(self, deadline: float) -> dict[str, Any]:
        assert self.reader is not None
        remaining = deadline - time.monotonic()
        require(remaining > 0, "group control deadline")
        line = await asyncio.wait_for(self.reader.readline(), remaining)
        return decode(line, self.binding)

    async def receive_hello(self) -> None:
        value = await self.read(self.binding["work_deadline"])
        assert self.process is not None
        require(
            value.get("event") == "hello"
            and value.get("pid") == self.process.pid
            and value.get("ppid") == os.getpid()
            and value.get("process_group") == self.process.pid,
            "worker hello ownership differs",
        )
        self.hello = value

    async def lease(self, stage: str) -> dict[str, Any]:
        require(self.ordinal < len(STAGES) and stage == STAGES[self.ordinal], "coordinator stage reordered")
        assert self.writer is not None
        lease = uuid.uuid4().hex
        started = time.monotonic()
        remaining = MAX_TRIAL_SECONDS - self.active_s
        limit = min(remaining, PHASE_LIMIT_S) if stage.startswith("C") else remaining
        deadline = min(self.binding["work_deadline"], started + limit)
        require(deadline > started, "worker cumulative active budget exhausted")
        request = envelope(self.binding, command="grant", stage=stage, ordinal=self.ordinal, lease=lease)
        self.writer.write((json.dumps(request, allow_nan=False) + "\n").encode())
        await asyncio.wait_for(self.writer.drain(), min(5, deadline - time.monotonic()))
        event = {
            "stage": stage,
            "ordinal": self.ordinal,
            "lease": lease,
            "started": started,
            "deadline": deadline,
            "active_before_s": self.active_s,
        }
        self.events.append(event)
        self.finishing = stage == "finish"
        try:
            value = await self.read(deadline)
            require(
                value.get("event") == "ack"
                and value.get("stage") == stage
                and value.get("ordinal") == self.ordinal
                and value.get("lease") == lease,
                "stale/reordered group ACK",
            )
            require(started <= value["monotonic_s"] <= time.monotonic(), "worker ACK clock differs")
            if stage == "prepare":
                require(value["actual_arm"] == self.binding["actual_arm"], "prepared configuration differs")
                require(
                    value["manager_present"] is (self.binding["actual_arm"] == "R"), "prepared LoRA manager differs"
                )
                sdk = value["sdk_identity"]
                assert self.process is not None
                require(
                    sdk["ppid"] == self.process.pid and sdk["process_group"] == self.process.pid, "SDK lineage differs"
                )
                require(
                    sdk["source"] == self.binding["source_root"] and not sdk["forbidden_imports_present"],
                    "SDK source differs",
                )
                config_path = Path(self.binding["output"]) / "models/model.yaml"
                require(digest(config_path) == value["config_sha256"], "prepared config hash differs")
            if stage != "finish":
                require(value["quiet"]["pending"] == 0 and value["quiet"]["in_flight"] == 0, "worker ACK was not quiet")
            if stage.startswith("C"):
                concurrency = int(stage[1:])
                record = value["record"]
                require(
                    record["concurrency"] == concurrency and record["path"] == f"phase-C{concurrency}.json",
                    "ACK phase differs",
                )
                path = Path(self.binding["output"]) / "loadgen" / record["path"]
                # The worker verified the full hash before ACK. Keep only
                # metadata here; even the coordinator's redundant full-file
                # hash read waits until all nine timers have ended.
                require(
                    path.stat().st_size == record["bytes"],
                    "phase persistence differs",
                )
            event["ack"] = value
            self.ordinal += 1
            return value
        finally:
            event["ended"] = time.monotonic()
            self.active_s += event["ended"] - started
            event["active_total_s"] = self.active_s
            if self.active_s > MAX_TRIAL_SECONDS:
                error = sys.exception()
                if error is None:
                    raise AssertionError("worker cumulative active budget exceeded")
                error.add_note("Worker cumulative active budget also exceeded")


async def cleanup(peers: list[Peer], hard_deadline: float) -> list[dict[str, Any]]:
    """One shared deadline, all signals first; no per-child multiplication."""
    deadline = min(hard_deadline, time.monotonic() + GROUP_CLEANUP_S)
    for peer in peers:
        peer.stopping = True
        if peer.writer is not None:
            peer.writer.close()  # sole parent endpoint; explicit worker EOF cancellation
        if peer.process is not None:
            try:
                os.killpg(peer.process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
    processes = [peer.process for peer in peers if peer.process is not None]
    waits = [asyncio.create_task(process.wait()) for process in processes]
    if waits:
        _, pending = await asyncio.wait(waits, timeout=max(0, min(35, deadline - time.monotonic() - 5)))
        for process in processes:
            # A worker may have exited while a descendant remains. Signal its
            # entire owned group, not only PIDs whose returncode is still None.
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        if pending:
            _, pending = await asyncio.wait(pending, timeout=max(0, deadline - time.monotonic()))
        require(not pending, "owned worker reap exceeded shared cleanup deadline")
    receipt = []
    for peer in peers:
        if peer.process is not None:
            receipt.append(
                {
                    "role": peer.binding["logical_role"],
                    "pid": peer.process.pid,
                    "returncode": peer.process.returncode,
                    "reaped": peer.process.returncode is not None,
                }
            )
        peer.log.close()
    return receipt


async def watch_peer(peer: Peer, owner: asyncio.Task) -> None:
    assert peer.process is not None
    await peer.process.wait()
    if not peer.finishing and not peer.stopping:
        owner.cancel(f"worker {peer.binding['logical_role']} exited before group finish")


def verify_phase_files(peers: list[Peer]) -> None:
    require(
        len(peers) == len(ORDERS[0]) and all(peer.ordinal == len(STAGES) - 1 for peer in peers),
        "group timers incomplete",
    )
    for peer in peers:
        for event in peer.events[1:]:
            record = event["ack"]["record"]
            path = Path(peer.binding["output"]) / "loadgen" / record["path"]
            require(
                path.stat().st_size == record["bytes"] and digest(path) == record["sha256"], "phase persistence differs"
            )


async def join_cleanup(task: asyncio.Task, primary: BaseException | None) -> Any:
    """Repeated coordinator TERM/cancellation cannot abandon owned teardown.

    cleanup itself owns the one absolute deadline. This join does not reset it
    and does not cancel the shared teardown task when its caller is cancelled.
    """
    cancelled = None
    while not task.done():
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError as error:
            cancelled = error
        except BaseException:  # noqa: BLE001 -- retrieve the same task exception below
            break  # retrieve the completed task's original exception below
    result = task.result()
    if cancelled is not None:
        if primary is None:
            raise cancelled
        primary.add_note("Coordinator cancellation during cleanup retained after owned reaping")
    return result


async def run_group(args: argparse.Namespace, manifest: dict[str, Any], pair: int) -> dict[str, Any]:
    started = time.monotonic()
    hard_deadline = started + args.group_work_s + GROUP_CLEANUP_S
    group: dict[str, Any] = {
        "pair": pair,
        "order": list(ORDERS[pair]),
        "started": started,
        "work_deadline": started + args.group_work_s,
        "hard_deadline": hard_deadline,
        "completed_valid": False,
        "leases": [],
        "trials": [],
    }
    peers: list[Peer] = []
    watchers: list[asyncio.Task] = []
    owner = asyncio.current_task()
    assert owner is not None
    nonce = uuid.uuid4().hex
    try:
        for role in ORDERS[pair]:
            binding = binding_for(args, manifest, pair, role, nonce, started)
            path = args.output / f"binding-{pair}-{role}.json"
            atomic_json(path, binding)
            peer = Peer(binding, (args.output / f"trial-{pair}-{role}.log").open("wb"))
            peers.append(peer)
            await peer.start(args, path)
            watchers.append(asyncio.create_task(watch_peer(peer, owner)))
        for peer in peers:
            await peer.receive_hello()
        for stage in STAGES:
            # Phase-major scheduling, and finalization only after all nine
            # persisted phase ACKs. No background task grants another lease.
            if stage == "finish":
                verify_phase_files(peers)
                group["phase_hashes_verified_at"] = time.monotonic()
            for peer in peers:
                await peer.lease(stage)
                group["leases"].append({"role": peer.binding["logical_role"], **peer.events[-1]})
                atomic_json(args.output / f"group-{pair}.json", group)
                if stage == "finish":
                    assert peer.process is not None
                    await asyncio.wait_for(peer.process.wait(), max(0.001, min(10, hard_deadline - time.monotonic())))
                    require(peer.process.returncode == 0, "finalized worker exited unsuccessfully")
        # Reopen raw trial payloads only after every timer, every finish grant,
        # SDK reap and server lifecycle check in the triplet is complete.
        for peer in peers:
            path = Path(peer.binding["output"]) / "trial.json"
            require(digest(path) == peer.events[-1]["ack"]["terminal_sha256"], "worker terminal hash differs")
            trial = json.loads(path.read_text())
            require(
                trial.get("group") == peer.binding and trial.get("actual_arm") == peer.binding["actual_arm"],
                "terminal binding differs",
            )
            require(trial["pair"] == pair and trial["arm"] == peer.binding["logical_role"], "terminal role differs")
            require(trial["passed"] and trial["shutdown_complete"], "worker correctness/lifecycle failed")
            trial.update(
                source_commit=peer.binding["source_commit"],
                process_returncode=0,
                terminal_record=str(path.relative_to(args.output)),
                terminal_record_sha256=digest(path),
            )
            group["trials"].append(trial_summary(trial))
        group["completed_valid"] = True
    except BaseException as error:
        group["error"] = {"type": type(error).__name__, "message": str(error)}
        raise
    finally:
        primary = sys.exception()
        task = asyncio.create_task(cleanup(peers, hard_deadline))
        try:
            group["cleanup"] = await join_cleanup(task, primary)
        except BaseException as error:
            group["completed_valid"] = False
            group["cleanup_error"] = {"type": type(error).__name__, "message": str(error)}
            if primary is None:
                raise
            primary.add_note(f"Group cleanup failed: {type(error).__name__}: {error}")
        finally:
            for watcher in watchers:
                watcher.cancel()
            await asyncio.gather(*watchers, return_exceptions=True)
            group["peer_events"] = {peer.binding["logical_role"]: peer.events for peer in peers}
            group["finished"] = time.monotonic()
            try:
                atomic_json(args.output / f"group-{pair}.json", group)
            except BaseException as error:
                if primary is None:
                    raise
                primary.add_note(f"Group receipt failed: {type(error).__name__}: {error}")
    return group


def validate_trials(trials: list[dict[str, Any]], mode: str, *, smoke: bool) -> None:
    schedule = (
        [(0, role) for role in ORDERS[0]] if smoke else [(i, role) for i, order in enumerate(ORDERS) for role in order]
    )
    require([(t["pair"], t["arm"]) for t in trials] == schedule, "group trial schedule incomplete")
    for trial in trials:
        binding = trial["group"]
        validate_binding(binding)
        require(
            binding["mode"] == mode and binding["logical_role"] == trial["arm"] and binding["pair"] == trial["pair"],
            "trial group binding differs",
        )
        require(trial["actual_arm"] == selection(mode, trial["arm"])[0], "actual trial configuration differs")
        require(trial["manager_present"] is (trial["actual_arm"] == "R"), "trial manager differs")
        require(trial["passed"] and trial["shutdown_complete"], "group trial failed")
        require(
            trial["environment"]["qualification"] is (mode == "qualification" and not smoke),
            "trial qualification mode differs",
        )


async def run(args: argparse.Namespace) -> dict[str, Any]:
    require(args.mode in MODES and 0 < args.deadline_s <= RUNNER_LIMIT_S, "invalid interleaved mode/deadline")
    require(0 < args.group_work_s <= GROUP_WORK_S, "invalid group work deadline")
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    deadline = started + args.deadline_s
    summary: dict[str, Any] = {
        "schema": 3,
        "mode": args.mode,
        "schedule": "whole-phase",
        "smoke": args.smoke,
        "completed_valid": False,
        "qualification_passed": False,
        "passed": False,
        "orders": ORDERS,
        "groups": [],
        "trials": [],
        "admissions": [],
        "harness_manifest": harness_manifest(),
        "runner_clock": {
            "monotonic_start_s": started,
            "wall_start_ns": time.time_ns(),
            "deadline_monotonic_s": deadline,
            "budget_s": args.deadline_s,
        },
    }
    try:
        manifest = extract(args.inputs.resolve(), args.output / "sources")
        summary["source_manifest"] = manifest
        if args.environment_receipt is not None:
            summary["environment_preparation"] = {
                "sha256": digest(args.environment_receipt),
                "receipt": json.loads(args.environment_receipt.read_text()),
            }
        for pair in range(1 if args.smoke else len(ORDERS)):
            check = {"pair": pair, **admission(time.monotonic(), deadline, args.group_work_s)}
            summary["admissions"].append(check)
            require(check["admitted"], "insufficient time for complete bounded group and shared cleanup")
            print(f"Starting whole-phase group {pair + 1} ({args.mode})", flush=True)
            group = await run_group(args, manifest, pair)
            summary["groups"].append(
                {
                    "pair": pair,
                    "completed_valid": group["completed_valid"],
                    "record": f"group-{pair}.json",
                    "sha256": digest(args.output / f"group-{pair}.json"),
                }
            )
            summary["trials"].extend(group["trials"])
            atomic_json(args.output / "summary.json", summary)
            print(f"Finished whole-phase group {pair + 1}", flush=True)
        trials = [load_trial(args.output, record) for record in summary["trials"]]
        validate_trials(trials, args.mode, smoke=args.smoke)
        # Smoke still validates all three peers' numerics/memory: fill only the
        # iteration count via a dedicated single-triplet check, not fabricated
        # replicas. Full acceptance always uses the original six-triplet check.
        if args.smoke:
            summary["correctness_memory"] = smoke_cross_checks(trials)
        else:
            summary["correctness_memory"] = cross_checks(trials)
            summary["statistics"] = compare(trials)
        summary["completed_valid"] = True
        summary["qualification_passed"] = (
            not args.smoke and args.mode == "qualification" and summary["statistics"]["passed"]
        )
        summary["passed"] = (
            summary["completed_valid"]
            if args.mode == "baseline-pilot" or args.smoke
            else summary["qualification_passed"]
        )
        require(summary["passed"], "fixed non-inferiority gates did not pass")
    except BaseException as error:
        summary["error"] = {"type": type(error).__name__, "message": str(error)}
        raise
    finally:
        atomic_json(args.output / "summary.json", summary)
    return summary


def smoke_cross_checks(trials: list[dict[str, Any]]) -> dict[str, Any]:
    # Reuse the exact peer checks without weakening them or inventing six
    # observations. The helper reads ORDERS only to determine the pair count.
    check = FunctionType(cross_checks.__code__, {**cross_checks.__globals__, "ORDERS": ORDERS[:1]})
    return check(trials)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--fixtures", type=Path, required=True)
    parser.add_argument("--snapshot", type=Path)
    parser.add_argument("--environment-receipt", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", choices=MODES, required=True)
    parser.add_argument("--deadline-s", type=float, default=RUNNER_LIMIT_S)
    parser.add_argument("--group-work-s", type=float, default=GROUP_WORK_S)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    async def launch() -> None:
        task = asyncio.current_task()
        assert task is not None
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGTERM, signal.SIGINT):
            loop.add_signal_handler(sig, task.cancel)
        await run(args)

    asyncio.run(launch())


if __name__ == "__main__":
    main()
