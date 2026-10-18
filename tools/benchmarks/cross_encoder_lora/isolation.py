"""Async parent ownership of one exec-only SDK child; no compute provisioning."""

from __future__ import annotations

import asyncio
import gc
import json
import math
import os
import sys
import time
import uuid
from pathlib import Path
from types import TracebackType
from typing import Any, Self

from tools.benchmarks.cross_encoder_lora.evidence import Evidence, atomic_json, digest
from tools.benchmarks.cross_encoder_lora.loadgen import CONTROL_LIMIT, CONTROL_TIMEOUT_S, PHASES
from tools.benchmarks.cross_encoder_lora.process_metrics import (
    boundary_gc_report,
    bracket,
    snapshot,
    validate_boundary_gc,
)
from tools.benchmarks.cross_encoder_lora.protocol import PERFORMANCE_FUTURE_TIMEOUT_S, require


class IsolatedLoadgen:
    """Leave uvicorn running while awaiting a bounded, same-group SDK child."""

    def __init__(
        self, url: str, source: Path, evidence: Evidence, *, smoke: bool, idle_deadline: float | None = None
    ) -> None:
        self.url, self.source, self.evidence, self.smoke = url, source.resolve(), evidence, smoke
        self.idle_deadline = idle_deadline
        self.token = uuid.uuid4().hex
        self.process: asyncio.subprocess.Process | None = None
        self.log: Any = None
        self.records: list[dict[str, Any]] = []
        self.observations: dict[int, dict[str, Any]] = {}
        self.receipt: dict[str, Any] = {"passed": False, "reaped": False, "phases": self.records}
        self.exit_watcher: asyncio.Task | None = None
        self.expected_exit = False

    async def _send(self, command: str, concurrency: int | None = None) -> None:
        assert self.process is not None
        assert self.process.stdin is not None
        value = {"token": self.token, "command": command, "concurrency": concurrency}
        self.process.stdin.write((json.dumps(value) + "\n").encode())
        await asyncio.wait_for(self.process.stdin.drain(), timeout=5)

    async def _read(
        self, event: str, concurrency: int | None = None, *, wait_s: float = CONTROL_TIMEOUT_S
    ) -> dict[str, Any]:
        assert self.process is not None
        assert self.process.stdout is not None
        line = await asyncio.wait_for(self.process.stdout.readline(), timeout=wait_s)
        require(
            bool(line) and len(line) <= CONTROL_LIMIT and line.endswith(b"\n"), "missing or oversized loadgen message"
        )
        value = json.loads(line)
        require(isinstance(value, dict), "invalid loadgen control object")
        require(value.get("token") == self.token and value.get("event") == event, "unexpected loadgen event")
        require(value.get("concurrency") == concurrency, "loadgen phase identity differs")
        return value

    async def __aenter__(self) -> Self:
        command = [
            sys.executable,
            "-m",
            "tools.benchmarks.cross_encoder_lora.loadgen",
            "--source",
            str(self.source),
            "--output",
            str(self.evidence.directory / "loadgen"),
            "--url",
            self.url,
            "--token",
            self.token,
        ]
        if self.smoke:
            command.append("--smoke")
        if self.idle_deadline is not None:
            require(
                math.isfinite(self.idle_deadline) and self.idle_deadline > time.monotonic(), "invalid idle deadline"
            )
            command.extend(["--idle-deadline", str(self.idle_deadline)])
        self.log = (self.evidence.directory / "loadgen.log").open("wb")
        try:
            self.process = await asyncio.create_subprocess_exec(
                *command,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=self.log,
                close_fds=True,
                limit=CONTROL_LIMIT,
            )
            hello = await self._read("hello")
            identity = hello["identity"]
            require(identity["pid"] == self.process.pid and identity["pid"] != os.getpid(), "loadgen PID differs")
            require(
                identity["ppid"] == os.getpid() and identity["process_group"] == os.getpgrp(),
                "loadgen ownership/group differs",
            )
            require(
                identity["source"] == str(self.source) and identity["forbidden_imports_present"] is False,
                "loadgen source/import preflight failed",
            )
            require(hello.get("idle_deadline") == self.idle_deadline, "loadgen idle deadline differs")
            sdk_path = Path(identity["sdk_path"])
            require(
                sdk_path.is_relative_to(self.source / "packages/sie_sdk/src"), "loadgen SDK source escaped its archive"
            )
            require(digest(sdk_path) == identity["sdk_sha256"], "loadgen SDK implementation hash differs")
            clock = time.get_clock_info("perf_counter")
            require(
                identity["clock"]["implementation"] == clock.implementation,
                "client/server monotonic clock implementations differ",
            )
            self.receipt["identity"] = identity
            self.evidence.emit("loadgen_started", **identity)
            if self.idle_deadline is not None:
                owner = asyncio.current_task()
                assert owner is not None
                self.exit_watcher = asyncio.create_task(self._watch_group_exit(owner))
            return self
        except BaseException as error:
            try:
                await self._cleanup(False)
            except BaseException as cleanup_error:  # noqa: BLE001 -- retain primary startup failure
                error.add_note(f"Loadgen startup cleanup failed: {type(cleanup_error).__name__}: {cleanup_error}")
            raise

    async def _watch_group_exit(self, owner: asyncio.Task) -> None:
        assert self.process is not None
        await self.process.wait()
        if not self.expected_exit:
            owner.cancel("group SDK child exited before finalization")

    async def phase(self, concurrency: int) -> dict[str, Any]:
        require(
            len(self.records) < len(PHASES) and concurrency == PHASES[len(self.records)],
            "duplicate or reordered measured phase",
        )
        observation: dict[str, Any] = {"server_pid": os.getpid(), "client_identity": self.receipt["identity"]}
        try:
            await self._send("phase", concurrency)
            await self._read("ready", concurrency)
            callbacks = tuple(gc.callbacks)
            observation["before"] = snapshot()
            await self._send("go", concurrency)
            measured = await self._read("measured", concurrency, wait_s=PERFORMANCE_FUTURE_TIMEOUT_S)
            clock = measured["clock"]
            require(
                set(clock) == {"start", "deadline", "end"}
                and all(type(v) in (int, float) and math.isfinite(v) for v in clock.values()),
                "invalid loadgen measurement clock",
            )
            require(
                0 < clock["start"] < clock["end"] and clock["deadline"] > clock["start"], "loadgen clock order differs"
            )
            observation["clock"] = clock
            observation["after"] = snapshot()
            observation["gc"] = boundary_gc_report(observation["before"], observation["after"], callbacks)
            observation["resources"] = bracket(observation["before"], observation["after"], clock)
            await self._send("observed", concurrency)
            complete = await self._read("phase_complete", concurrency)
            record = complete["record"]
            require(
                type(record["concurrency"]) is int
                and record["concurrency"] == concurrency
                and record["path"] == f"phase-C{concurrency}.json",
                "loadgen record identity differs",
            )
            path = self.evidence.directory / "loadgen" / record["path"]
            require(
                path.is_file() and path.stat().st_size == record["bytes"] and digest(path) == record["sha256"],
                "loadgen phase record is incomplete or corrupt",
            )
            self.records.append(record)
            self.observations[concurrency] = observation
            self.evidence.emit(
                "loadgen_phase_complete", concurrency=concurrency, record=record, server_observation=observation
            )
            return record
        finally:
            primary_error = sys.exception()
            try:
                atomic_json(self.evidence.directory / f"server-C{concurrency}-observations.json", observation)
            except BaseException as write_error:
                if primary_error is None:
                    raise
                primary_error.add_note(
                    f"Secondary server observation write failure: {type(write_error).__name__}: {write_error}"
                )

    async def _cleanup(self, successful: bool) -> None:
        self.expected_exit = True
        try:
            if self.process is not None:
                try:
                    if successful:
                        require(len(self.records) == len(PHASES), "loadgen did not complete every phase")
                        await self._send("finish")
                        finished = await self._read("finished")
                        await asyncio.wait_for(self.process.wait(), timeout=10)
                        require(self.process.returncode == 0, "loadgen failed after final evidence")
                        terminal = self.evidence.directory / "loadgen/terminal.json"
                        require(digest(terminal) == finished["terminal_sha256"], "loadgen terminal record differs")
                        self._validate_terminal(json.loads(terminal.read_text()))
                        self.receipt["terminal_sha256"] = finished["terminal_sha256"]
                        self.receipt["passed"] = True
                finally:
                    if self.process.returncode is None:
                        try:
                            self.process.terminate()
                        except ProcessLookupError:
                            self.receipt["termination_raced_exit"] = True
                        try:
                            await asyncio.wait_for(self.process.wait(), timeout=3)
                        except TimeoutError:
                            try:
                                self.process.kill()
                            except ProcessLookupError:
                                self.receipt["kill_raced_exit"] = True
                            await asyncio.wait_for(self.process.wait(), timeout=2)
                    self.receipt["returncode"] = self.process.returncode
                    self.receipt["reaped"] = self.process.returncode is not None
                    if self.process.stdin is not None:
                        self.process.stdin.close()
            else:
                self.receipt["reaped"] = True
        finally:
            if self.exit_watcher is not None:
                self.exit_watcher.cancel()
                await asyncio.gather(self.exit_watcher, return_exceptions=True)
            if self.log is not None:
                self.log.close()
            atomic_json(self.evidence.directory / "loadgen-process.json", self.receipt)

    async def __aexit__(
        self, error_type: type[BaseException] | None, error: BaseException | None, traceback: TracebackType | None
    ) -> None:
        task = asyncio.create_task(self._cleanup(error_type is None))
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError as cancelled:
            # Cancellation may arrive while the successful finish handshake is
            # waiting for a silent child. Cancel that wait, join its bounded
            # teardown, then explicitly reap even if it never started running.
            task.cancel()
            try:
                await asyncio.shield(task)
            except asyncio.CancelledError:
                pass
            except BaseException as cleanup_error:  # noqa: BLE001 -- cancellation remains primary
                self.receipt["cancelled_cleanup_error"] = {
                    "type": type(cleanup_error).__name__,
                    "message": str(cleanup_error),
                }
            try:
                await asyncio.shield(asyncio.create_task(self._cleanup(False)))
            except BaseException as cleanup_error:  # noqa: BLE001 -- preserve cancellation after bounded reaping
                cancelled.add_note(
                    f"Secondary cancelled cleanup failure: {type(cleanup_error).__name__}: {cleanup_error}"
                )
            raise
        except BaseException as cleanup_error:
            if error is None:
                raise
            error.add_note(f"Secondary loadgen cleanup failure: {type(cleanup_error).__name__}: {cleanup_error}")
            self.receipt["cleanup_error"] = {"type": type(cleanup_error).__name__, "message": str(cleanup_error)}
            try:
                atomic_json(self.evidence.directory / "loadgen-process.json", self.receipt)
            except BaseException as write_error:  # noqa: BLE001 -- original phase failure remains primary
                error.add_note(f"Secondary cleanup receipt write failure: {type(write_error).__name__}: {write_error}")

    def _validate_terminal(self, terminal: dict[str, Any]) -> None:
        require(terminal.get("passed") is True and "error" not in terminal, "loadgen terminal did not pass")
        require(terminal.get("identity") == self.receipt["identity"], "loadgen terminal identity differs")
        require(terminal.get("smoke") is self.smoke, "loadgen terminal mode differs")
        require(terminal.get("idle_deadline") == self.idle_deadline, "loadgen terminal idle deadline differs")
        require(terminal.get("phases") == self.records, "loadgen terminal phase records differ")
        require(tuple(record["concurrency"] for record in self.records) == PHASES, "loadgen terminal schedule differs")
        require(
            type(terminal.get("finished_ns")) is int and terminal["finished_ns"] > 0,
            "loadgen terminal clock is missing",
        )

    def _validate_phase(self, record: dict[str, Any], value: dict[str, Any]) -> dict[str, Any]:
        concurrency = record["concurrency"]
        server = self.observations[concurrency]
        identity = self.receipt["identity"]
        require(
            type(value.get("concurrency")) is int and value["concurrency"] == concurrency,
            "phase payload concurrency differs",
        )
        require(
            value.get("identity") == identity and value.get("smoke") is self.smoke, "phase payload provenance differs"
        )
        require(value.get("measurement_clock") == server["clock"], "phase payload differs from observed IPC clock")
        client = value["client_observation"]
        require(client.get("clock") == server["clock"], "client observation clock differs")
        require(
            all(client[key]["core"]["pid"] == identity["pid"] for key in ("before", "after")),
            "phase observation PID differs",
        )
        require(client["gc"]["pid"] == identity["pid"], "client GC observation PID differs")
        require(server["gc"]["pid"] == os.getpid(), "server GC observation PID differs")
        for observation in (client, server):
            validate_boundary_gc(observation)
        value["server_observation"] = server
        value["combined_estimated_observer_fraction"] = None
        value["observer_cost_gate_applied"] = False
        value["observer_cost_gate_status"] = "not_applicable"
        return value

    def results(self) -> dict[str, dict[str, Any]]:
        require(
            self.receipt["passed"] and self.receipt["reaped"],
            "loadgen must finish and be reaped before ingesting raw rows",
        )
        values = {}
        for record in self.records:
            path = self.evidence.directory / "loadgen" / record["path"]
            require(
                path.stat().st_size == record["bytes"] and digest(path) == record["sha256"],
                "completed loadgen evidence changed",
            )
            value = self._validate_phase(record, json.loads(path.read_text()))
            values[str(record["concurrency"])] = value
        return values
