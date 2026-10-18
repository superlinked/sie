"""One SDK-only exec child, reused across a trial's three measured phases."""

from __future__ import annotations

import argparse
import gc
import inspect
import json
import math
import os
import queue
import select
import signal
import sys
import threading
import time
from pathlib import Path
from typing import Any

from sie_sdk import SIEClient

from tools.benchmarks.cross_encoder_lora.client import performance
from tools.benchmarks.cross_encoder_lora.evidence import Evidence, atomic_json, digest
from tools.benchmarks.cross_encoder_lora.process_metrics import (
    boundary_gc_report,
    bracket,
    snapshot,
    validate_boundary_gc,
)
from tools.benchmarks.cross_encoder_lora.protocol import require

CONTROL_LIMIT = 65_536
CONTROL_TIMEOUT_S = 30
PHASES = (1, 4, 16)
FORBIDDEN_IMPORTS = ("torch", "transformers", "peft", "sie_server")


def require_sdk_only() -> None:
    require(
        not any(name.split(".", 1)[0] in FORBIDDEN_IMPORTS for name in sys.modules),
        "load generator imported serving or CUDA model libraries",
    )


class Control:
    """One stdin reader; loss of its sole parent writer ends this owned child."""

    def __init__(self, token: str) -> None:
        self.token = token
        self.messages: queue.Queue[dict[str, Any]] = queue.Queue(maxsize=8)
        self.normal_exit = False
        self.stop_read, self.stop_write = os.pipe()
        self.reader = threading.Thread(target=self._read, name="loadgen-parent-pipe", daemon=True)
        self.reader.start()

    def _read(self) -> None:
        pending = bytearray()
        try:
            while True:
                readable, _, _ = select.select([sys.stdin.fileno(), self.stop_read], [], [])
                if self.stop_read in readable:
                    return
                data = os.read(sys.stdin.fileno(), 4096)
                if not data:
                    if not self.normal_exit:
                        # A hard-killed parent cannot finalize current samples.
                        # Previously completed atomic files remain the evidence.
                        os.kill(os.getpid(), signal.SIGTERM)
                    return
                pending.extend(data)
                while b"\n" in pending:
                    end = pending.index(b"\n") + 1
                    line = bytes(pending[:end])
                    del pending[:end]
                    try:
                        require(len(line) <= CONTROL_LIMIT, "invalid control line length")
                        value = json.loads(line)
                        require(isinstance(value, dict), "invalid control object")
                    except (ValueError, AssertionError) as error:
                        value = {"reader_error": type(error).__name__}
                    # Malformed input does not disable the sole EOF monitor.
                    # A flooded queue fails closed without blocking this reader.
                    self.messages.put_nowait(value)
                require(len(pending) <= CONTROL_LIMIT, "unterminated control line is too large")
        except BaseException:  # noqa: BLE001 -- losing the lifetime monitor must terminate this owned child
            if not self.normal_exit:
                os.kill(os.getpid(), signal.SIGTERM)

    def close(self) -> None:
        # Raw-fd reads avoid a daemon holding Python's buffered stdin lock at
        # interpreter shutdown. Wake and join it explicitly before normal exit.
        self.normal_exit = True
        os.write(self.stop_write, b"x")
        self.reader.join(timeout=2)
        require(not self.reader.is_alive(), "parent-pipe monitor failed to stop")
        os.close(self.stop_read)
        os.close(self.stop_write)

    def receive(
        self, command: str, concurrency: int | None = None, *, idle_deadline: float | None = None
    ) -> dict[str, Any]:
        wait_s = CONTROL_TIMEOUT_S
        if idle_deadline is not None:
            require(command in {"phase", "finish"}, "group idle deadline used for active handshake")
            require(math.isfinite(idle_deadline), "invalid group idle deadline")
            wait_s = idle_deadline - time.monotonic()
            require(wait_s > 0, "group idle deadline expired")
        value = self.messages.get(timeout=wait_s)
        require("reader_error" not in value, "control reader failed")
        require(value.get("token") == self.token and value.get("command") == command, "unexpected parent command")
        require(value.get("concurrency") == concurrency, "parent phase identity differs")
        return value

    def send(self, event: str, concurrency: int | None = None, **fields: Any) -> None:
        value = {"token": self.token, "event": event, "concurrency": concurrency, **fields}
        line = json.dumps(value, allow_nan=False)
        require(len(line.encode()) < CONTROL_LIMIT, "control output exceeds bounded protocol")
        print(line, flush=True)


def identity(source: Path) -> dict[str, Any]:
    require_sdk_only()
    implementation = Path(inspect.getfile(SIEClient)).resolve()
    expected = source.resolve() / "packages/sie_sdk/src"
    require(implementation.is_relative_to(expected), "load generator SDK did not come from the selected source archive")
    clock = time.get_clock_info("perf_counter")
    require(clock.monotonic and not clock.adjustable, "load generator requires a monotonic performance clock")
    return {
        "pid": os.getpid(),
        "ppid": os.getppid(),
        "process_group": os.getpgrp(),
        "sdk_path": str(implementation),
        "sdk_sha256": digest(implementation),
        "python": sys.executable,
        "source": str(source.resolve()),
        "clock": {name: getattr(clock, name) for name in ("implementation", "resolution", "monotonic", "adjustable")},
        "forbidden_imports_present": False,
    }


def run(args: argparse.Namespace) -> None:
    evidence = Evidence(args.output.resolve())
    control = Control(args.token)
    terminal: dict[str, Any] = {"passed": False, "phases": [], "identity": identity(args.source), "smoke": args.smoke}
    retained_results = []
    idle_deadline = getattr(args, "idle_deadline", None)
    terminal["idle_deadline"] = idle_deadline
    control.send("hello", identity=terminal["identity"], idle_deadline=idle_deadline)
    try:
        for concurrency in PHASES:
            if idle_deadline is None:
                control.receive("phase", concurrency)
            else:
                control.receive("phase", concurrency, idle_deadline=idle_deadline)
            require_sdk_only()
            callbacks: tuple[Any, ...] = ()
            observations: dict[str, Any] = {}

            def before_measure() -> None:
                nonlocal callbacks
                callbacks = tuple(gc.callbacks)
                observations["before"] = snapshot()
                control.send("ready", concurrency)
                control.receive("go", concurrency)

            def after_measure(clock: dict[str, float]) -> None:
                observations["clock"] = clock
                observations["after"] = snapshot()
                observations["gc"] = boundary_gc_report(observations["before"], observations["after"], callbacks)
                control.send("measured", concurrency, clock=clock)
                control.receive("observed", concurrency)

            try:
                result = performance(
                    args.url,
                    concurrency,
                    evidence,
                    smoke=args.smoke,
                    before_measure=before_measure,
                    after_measure=after_measure,
                )
            finally:
                primary_error = sys.exception()
                try:
                    atomic_json(evidence.directory / f"phase-C{concurrency}-observations.json", observations)
                except BaseException as write_error:
                    if primary_error is None:
                        raise
                    primary_error.add_note(
                        f"Secondary client observation write failure: {type(write_error).__name__}: {write_error}"
                    )
            validate_boundary_gc(observations)
            observations["resources"] = bracket(
                observations["before"], observations["after"], result["measurement_clock"]
            )
            result["client_observation"] = observations
            result["identity"] = terminal["identity"]
            result["smoke"] = args.smoke
            require_sdk_only()
            path = evidence.directory / f"phase-C{concurrency}.json"
            atomic_json(path, result)
            record = {
                "concurrency": concurrency,
                "path": path.name,
                "sha256": digest(path),
                "bytes": path.stat().st_size,
            }
            terminal["phases"].append(record)
            atomic_json(evidence.directory / "terminal.json", terminal)
            # Preserve the historical request-history lifetime, but only in the
            # load generator's heap. The serving process never ingests these
            # rows until every timed phase is complete.
            retained_results.append(result)
            control.send("phase_complete", concurrency, record=record)
        if idle_deadline is None:
            control.receive("finish")
        else:
            control.receive("finish", idle_deadline=idle_deadline)
        terminal["passed"] = True
    except BaseException as error:
        terminal["error"] = {"type": type(error).__name__, "message": str(error)}
        raise
    finally:
        control.close()
        terminal["finished_ns"] = time.time_ns()
        atomic_json(evidence.directory / "terminal.json", terminal)
        if terminal["passed"]:
            control.send("finished", terminal_sha256=digest(evidence.directory / "terminal.json"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--url", required=True)
    parser.add_argument("--token", required=True)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--idle-deadline", type=float)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
