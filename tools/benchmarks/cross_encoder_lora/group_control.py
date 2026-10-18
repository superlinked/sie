"""Dedicated group control and parent-death ownership, outside request timing."""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import queue
import select
import signal
import socket
import sys
import threading
import time
from pathlib import Path
from typing import Any

from tools.benchmarks.cross_encoder_lora.evidence import atomic_json, digest, fingerprint
from tools.benchmarks.cross_encoder_lora.protocol import BASE_REVISION, MAX_TRIAL_SECONDS, ORDERS, require

LIMIT = 65_536
GROUP_WORK_S = 600
GROUP_CLEANUP_S = 90
PHASE_LIMIT_S = 120
RUNNER_LIMIT_S = 3900
STAGES = ("prepare", "C1", "C4", "C16", "finish")
MODES = ("baseline-pilot", "qualification")
NONCE_LENGTH = 32


def selection(mode: str, role: str) -> tuple[str, str]:
    require(mode in MODES and role in "BNR" and len(role) == 1, "invalid group mode/role")
    return ("B", "B") if mode == "baseline-pilot" or role == "B" else (role, "candidate")


def validate_binding(binding: dict[str, Any]) -> None:
    require(binding["schema"] == 1 and binding["schedule"] == "whole-phase", "invalid group binding schema")
    arm, archive = selection(binding["mode"], binding["logical_role"])
    require((binding["actual_arm"], binding["source_key"]) == (arm, archive), "role/config/source mismatch")
    require(type(binding["pair"]) is int and 0 <= binding["pair"] < len(ORDERS), "invalid group pair")
    require(isinstance(binding["nonce"], str) and len(binding["nonce"]) == NONCE_LENGTH, "invalid group nonce")
    require(type(binding["coordinator_pid"]) is int and binding["coordinator_pid"] > 0, "invalid coordinator PID")
    for name in ("work_deadline", "hard_deadline", "group_started"):
        require(type(binding[name]) in (float, int) and math.isfinite(binding[name]), "invalid group deadline")
    require(
        0 < binding["work_deadline"] - binding["group_started"] <= GROUP_WORK_S
        and 0 < binding["hard_deadline"] - binding["work_deadline"] <= GROUP_CLEANUP_S,
        "group deadlines exceed fixed bounds",
    )
    require(binding["active_limit_s"] == MAX_TRIAL_SECONDS, "active budget differs")
    require(binding["phase_limit_s"] == PHASE_LIMIT_S, "phase budget differs")
    if archive == "B":
        require(binding["source_commit"] == BASE_REVISION, "baseline commit differs")


def validate_source(binding: dict[str, Any], source: Path, arm: str, pair: int) -> None:
    """Called under the prepare lease, before preflight/model initialization."""
    validate_binding(binding)
    require(binding["actual_arm"] == arm and binding["pair"] == pair, "worker arguments differ from binding")
    require(str(source.resolve()) == binding["source_root"], "worker source path differs")
    manifest_path = Path(binding["source_manifest"])
    require(digest(manifest_path) == binding["source_manifest_sha256"], "source manifest changed")
    manifest = json.loads(manifest_path.read_text())
    require(manifest["schema"] == 1 and manifest["dependency_lock_identical"], "source manifest mode differs")
    record = manifest["sources"][binding["source_key"]]
    require(record["commit"] == binding["source_commit"], "source commit differs")
    require(record["sha256"] == binding["archive_sha256"], "source archive binding differs")
    archive = manifest_path.parent / record["archive"]
    require(archive.resolve().is_relative_to(manifest_path.parent.resolve()), "source archive escapes inputs")
    require(digest(archive) == record["sha256"], "source archive changed")
    for name, expected in binding["source_files"].items():
        path = source / name
        require(path.resolve().is_relative_to(source.resolve()) and digest(path) == expected, "source file changed")


def envelope(binding: dict[str, Any], **fields: Any) -> dict[str, Any]:
    return {"binding_sha256": fingerprint(binding), **fields}


def decode(line: bytes, binding: dict[str, Any]) -> dict[str, Any]:
    require(bool(line) and len(line) <= LIMIT and line.endswith(b"\n"), "missing/oversized group message")
    value = json.loads(line)
    require(isinstance(value, dict) and value.get("binding_sha256") == fingerprint(binding), "group identity differs")
    return value


class Member:
    """One raw-fd monitor exists before heavy worker imports or uvicorn handlers.

    Its only writer is the coordinator. EOF cancels the asyncio experiment;
    if teardown or import is stuck, one hard deadline kills this worker's own
    process group (including its SDK child). No model work occurs in this thread.
    """

    def __init__(self, fd: int, binding: dict[str, Any], output: Path) -> None:
        validate_binding(binding)
        require(os.getpgrp() == os.getpid(), "group worker must own its process group")
        require(os.getppid() == binding["coordinator_pid"], "coordinator lineage differs")
        self.binding, self.output = binding, output
        self.socket = socket.socket(fileno=fd)
        self.socket.set_inheritable(False)
        self.socket.settimeout(5)
        self.messages: queue.Queue[dict[str, Any]] = queue.Queue(maxsize=2)
        self.stopped = threading.Event()
        self.wake_read, self.wake_write = os.pipe()
        self.available: asyncio.Event | None = None
        self.failure: str | None = None
        self.loop: asyncio.AbstractEventLoop | None = None
        self.owner: asyncio.Task | None = None
        self.index = 0
        self.current: dict[str, Any] | None = None
        self.history: list[dict[str, Any]] = []
        self.reader = threading.Thread(target=self._monitor, name="group-coordinator", daemon=True)
        self.reader.start()
        self.send(event="hello", pid=os.getpid(), ppid=os.getppid(), process_group=os.getpgrp())

    def _fail(self, reason: str) -> None:
        self.failure = reason
        if self.loop is not None and self.owner is not None:
            try:
                self.loop.call_soon_threadsafe(self.owner.cancel, reason)
            except RuntimeError:
                # A loop that already closed must not disable the raw-fd
                # monitor's independent hard group kill.
                pass

    def _monitor(self) -> None:
        pending = bytearray()
        deadline = self.binding["hard_deadline"]
        try:
            while not self.stopped.is_set():
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                ready, _, _ = select.select([self.socket, self.wake_read], [], [], remaining)
                if self.wake_read in ready:
                    return
                if not ready:
                    continue
                data = self.socket.recv(4096)
                require(bool(data), "coordinator EOF")
                pending.extend(data)
                while b"\n" in pending:
                    end = pending.index(b"\n") + 1
                    line = bytes(pending[:end])
                    del pending[:end]
                    self.messages.put_nowait(decode(line, self.binding))
                    if self.loop is not None and self.available is not None:
                        self.loop.call_soon_threadsafe(self.available.set)
                require(len(pending) <= LIMIT, "unterminated group message")
        except BaseException as error:  # noqa: BLE001 -- losing lifetime monitoring must fail closed
            if not self.stopped.is_set():
                self._fail(f"{type(error).__name__}: {error}")
                deadline = min(deadline, time.monotonic() + GROUP_CLEANUP_S)
        if not self.stopped.is_set():
            self._fail(self.failure or "group hard deadline")
            if not self.stopped.wait(max(0, deadline - time.monotonic())):
                os.killpg(os.getpid(), signal.SIGKILL)

    def bind_task(self) -> None:
        self.loop = asyncio.get_running_loop()
        self.owner = asyncio.current_task()
        self.available = asyncio.Event()
        if not self.messages.empty():
            self.available.set()
        require(self.failure is None, self.failure or "coordinator lost during imports")

    def send(self, **fields: Any) -> None:
        line = (json.dumps(envelope(self.binding, **fields), allow_nan=False) + "\n").encode()
        require(len(line) <= LIMIT, "group output too large")
        self.socket.sendall(line)

    async def grant(self, stage: str) -> None:
        require(
            self.current is None and self.index < len(STAGES) and stage == STAGES[self.index], "group stage reordered"
        )
        assert self.available is not None
        while self.messages.empty():
            require(self.failure is None, self.failure or "group monitor failed")
            remaining = self.binding["work_deadline"] - time.monotonic()
            require(remaining > 0, "group idle deadline")
            self.available.clear()
            if self.messages.empty():
                await asyncio.wait_for(self.available.wait(), remaining)
        value = self.messages.get_nowait()
        require(value.get("command") == "grant" and value.get("stage") == stage, "unexpected group grant")
        require(value.get("ordinal") == self.index, "stale group grant")
        require(isinstance(value.get("lease"), str) and len(value["lease"]) == NONCE_LENGTH, "invalid group lease")
        require(time.monotonic() < self.binding["work_deadline"], "group work deadline")
        self.current = value

    def acknowledge(self, **fields: Any) -> None:
        require(self.current is not None, "group ACK without active lease")
        assert self.current is not None
        event = {
            "event": "ack",
            "stage": self.current["stage"],
            "ordinal": self.index,
            "lease": self.current["lease"],
            "monotonic_s": time.monotonic(),
            **fields,
        }
        self.history.append(event)
        atomic_json(self.output / "group-control.json", {"binding": self.binding, "events": self.history})
        self.send(**event)
        self.current = None
        self.index += 1

    def close(self) -> None:
        self.stopped.set()
        os.write(self.wake_write, b"x")
        self.reader.join(timeout=2)
        require(not self.reader.is_alive(), "group lifetime monitor did not stop")
        self.socket.close()
        os.close(self.wake_read)
        os.close(self.wake_write)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binding", type=Path, required=True)
    parser.add_argument("--control-fd", type=int, required=True)
    own, remainder = parser.parse_known_args()
    binding = json.loads(own.binding.read_text())
    member = Member(own.control_fd, binding, Path(binding["output"]))
    try:
        # Optional serving dependencies are imported only after the lifetime
        # monitor is live. Ordinary worker.py invocations do not use this path.
        from tools.benchmarks.cross_encoder_lora import worker  # noqa: PLC0415

        args = worker.arguments(remainder)
        args.group = member

        async def launch() -> None:
            member.bind_task()
            await worker.run(args)

        asyncio.run(launch())
    finally:
        primary = sys.exception()
        try:
            member.close()
        except BaseException as error:
            if primary is None:
                raise
            primary.add_note(f"Group monitor close failed: {type(error).__name__}: {error}")


if __name__ == "__main__":
    main()
