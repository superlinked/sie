"""Bounded, destructive-to-one-owned-child recovery probe; see README.md."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import importlib.metadata
import json
import logging
import math
import os
import socket
import time
from contextlib import ExitStack
from http import HTTPStatus
from pathlib import Path
from typing import Any

import httpx
import psutil
import uvicorn
from sie_sdk import SIEClient
from sie_server.app.app_factory import AppFactory
from sie_server.app.app_state_config import AppStateConfig

ROOT = Path(__file__).resolve().parents[3]
SURVIVOR = "Qwen/Qwen3Guard-Gen-4B"
VICTIM = "ibm-granite/granite-guardian-3.0-2b"
MODELS = [SURVIVOR, VICTIM]
PROMPT = "How can I make a cup of tea?"
MAX_DEADLINE_S = 1300
logger = logging.getLogger(__name__)


class Evidence:
    def __init__(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=False)
        self.path = path
        self.started = time.monotonic()
        self.rows: list[dict[str, Any]] = []

    def emit(self, event: str, **fields: Any) -> dict[str, Any]:
        row = {"event": event, "time": time.time(), "elapsed_s": time.monotonic() - self.started, **fields}
        self.rows.append(row)
        line = json.dumps(row, sort_keys=True)
        with (self.path / "events.jsonl").open("a") as output:
            output.write(line + "\n")
            output.flush()
            os.fsync(output.fileno())
        print("RECOVERY " + line, flush=True)
        return row

    def save(self, summary: dict[str, Any]) -> None:
        temporary = self.path / "summary.tmp"
        temporary.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
        temporary.replace(self.path / "summary.json")


class RequestProbe:
    def __init__(self, url: str, model: str, *, smoke: bool) -> None:
        self.model = model
        self.smoke = smoke
        self.responses: list[dict[str, Any]] = []
        self.request_started = time.monotonic()
        transport = httpx.Client(base_url=url, event_hooks={"response": [self.response]})
        self.client = SIEClient(url, api_key="", timeout_s=600, http_client=transport, remote="forbid")

    def response(self, response: httpx.Response) -> None:
        self.responses.append(
            {
                "status": response.status_code,
                "retry_after": response.headers.get("Retry-After"),
                "time": time.time(),
                "request_elapsed_s": time.monotonic() - self.request_started,
            }
        )

    def request(self, *, warm: bool = False) -> dict[str, Any]:
        self.responses = []
        started = time.monotonic()
        self.request_started = started
        error = None
        try:
            kwargs: dict[str, Any] = {
                "temperature": 0.0,
                "wait_for_capacity": warm,
                "max_oom_retries": 0,
                "provision_timeout_s": 600 if warm else 30,
                "read_timeout_s": 30,
            }
            if self.smoke:
                result = self.client.generate(self.model, PROMPT, max_new_tokens=32, **kwargs)
                text = result["text"]
            else:
                result = self.client.chat_completions(
                    self.model, [{"role": "user", "content": PROMPT}], max_tokens=32, **kwargs
                )
                text = result["choices"][0]["message"]["content"]
            if not isinstance(text, str) or not text.strip():
                raise RuntimeError("empty model output")
        except Exception as exc:  # noqa: BLE001 - Persist every request failure as measured evidence.
            error = {
                "type": type(exc).__name__,
                "message": str(exc)[:500],
                "code": getattr(exc, "code", None),
                "permanent": getattr(exc, "permanent", None),
            }
        return {
            "model": self.model,
            "duration_ms": (time.monotonic() - started) * 1000,
            "responses": self.responses,
            "error": error,
        }

    def close(self) -> None:
        self.client.close()


def snapshot(registry: Any) -> dict[str, Any]:
    models: dict[str, Any] = {}
    for name in MODELS:
        loaded = registry._loaded.get(name)
        process = getattr(loaded.adapter, "_process", None) if loaded else None
        failure = registry.get_failure(name)
        models[name] = {
            "loaded": registry.is_loaded(name),
            "pinned": registry._is_pinned(name),
            "pid": process.pid if process else None,
            "exit_code": process.poll() if process else None,
            "failure": None
            if failure is None
            else {
                "class": str(failure.error_class),
                "attempts": failure.attempts,
                "cooldown_s": failure.cooldown_s,
                "permanent": failure.is_permanent,
                "cooldown_remaining_s": None
                if failure.cooldown_s is None
                else max(0, failure.cooldown_s - (time.monotonic() - failure.last_attempt_ts)),
            },
        }
    return {
        "models": models,
        "resident_names": registry.loaded_model_names,
        "reserved_bytes": registry.loaded_model_memory_bytes,
    }


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for phase in ("before", "during", "after"):
        samples = [
            row for row in rows if row["event"] == "request" and row["model"] == SURVIVOR and row["phase"] == phase
        ]
        latencies = sorted(row["duration_ms"] for row in samples if row["error"] is None)
        result[phase] = {
            "count": len(samples),
            "errors": sum(row["error"] is not None for row in samples),
            "first_request_elapsed_s": samples[0]["elapsed_s"] - samples[0]["duration_ms"] / 1000 if samples else None,
            "last_response_elapsed_s": samples[-1]["elapsed_s"] if samples else None,
            "p50_ms": latencies[math.ceil(len(latencies) * 0.5) - 1] if latencies else None,
            "p95_ms": latencies[math.ceil(len(latencies) * 0.95) - 1] if latencies else None,
        }
    return result


async def run(args: argparse.Namespace, evidence: Evidence) -> bool:
    os.environ["SIE_TELEMETRY_DISABLED"] = "1"
    os.environ["SIE_DISK_CACHE_ENABLED"] = "false"
    os.environ["SIE_MODEL_LOAD_TIMEOUT_S"] = "600"
    os.environ["SIE_SGLANG_STARTUP_TIMEOUT_S"] = "550"
    ipc_path = Path(f"/tmp/sie-recovery-{os.getpid()}.sock")  # noqa: S108 - Unique owned local probe socket.
    if await asyncio.to_thread(ipc_path.exists):
        raise RuntimeError("refusing to replace an existing IPC socket")
    os.environ["SIE_IPC_SOCKET_PATH"] = str(ipc_path)
    versions = {}
    for name in ("torch", "transformers", "sglang"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    source_paths = [
        "packages/sie_server/src/sie_server/core/registry.py",
        "packages/sie_server/src/sie_server/adapters/sglang/_memory.py",
        "tools/benchmarks/model_recovery/probe.py",
        "packages/sie_server/models/Qwen__Qwen3Guard-Gen-4B.yaml",
        "packages/sie_server/models/ibm-granite__granite-guardian-3.0-2b.yaml",
    ]
    evidence.emit(
        "environment",
        smoke=args.smoke,
        versions=versions,
        source_sha256={path: hashlib.sha256((ROOT / path).read_bytes()).hexdigest() for path in source_paths},
        arguments=vars(args) | {"output": str(args.output)},
        pid=os.getpid(),
    )
    if not args.smoke:
        import torch  # noqa: PLC0415 - Optional GPU dependency.

        device = torch.cuda.get_device_properties(0)
        evidence.emit("gpu", name=device.name, total_bytes=device.total_memory)
        if "H100" not in device.name or device.total_memory < 75 * 1024**3:
            raise RuntimeError("this bounded protocol requires an H100 with at least 75 GiB")
    config = AppStateConfig(
        models_dir=ROOT / "packages/sie_server/models",
        device="cpu" if args.smoke else "cuda:0",
        model_filter=MODELS,
        pinned_models=[SURVIVOR],
        remote_serving=False,
        bind_host="127.0.0.1",
    )
    app = AppFactory.create_app(config)
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    server = uvicorn.Server(uvicorn.Config(app, log_level="warning", timeout_graceful_shutdown=45))
    serve_task = asyncio.create_task(server.serve(sockets=[sock]))
    clients: list[RequestProbe] = []
    child_identities: set[tuple[int, float]] = set()

    def remember_children(pid: int) -> None:
        child = psutil.Process(pid)
        for process in [child, *child.children(recursive=True)]:
            child_identities.add((process.pid, process.create_time()))

    registry = None
    passed = False
    cleanup_ok = False
    recovery_task = None
    try:
        async with asyncio.timeout(args.deadline_s):
            while not server.started:
                if serve_task.done():
                    await serve_task
                    raise RuntimeError("server exited before readiness")
                await asyncio.sleep(0.1)
            registry = app.state.registry
            evidence.emit("ready", state=snapshot(registry))
            remember_children(registry._loaded[SURVIVOR].adapter._process.pid)
            for name in MODELS:
                client = RequestProbe(f"http://127.0.0.1:{port}", name, smoke=args.smoke)
                clients.append(client)
                row = await asyncio.to_thread(client.request, warm=True)
                evidence.emit("warmup", **row, state=snapshot(registry))
                if row["error"]:
                    raise RuntimeError(f"warmup failed for {name}")
                pid = registry._loaded[name].adapter._process.pid
                remember_children(pid)
            survivor, victim = clients
            original_pid = registry._loaded[SURVIVOR].adapter._process.pid
            original_victim_pid = registry._loaded[VICTIM].adapter._process.pid
            if not registry._is_pinned(SURVIVOR):
                raise RuntimeError("supported pinned_models configuration did not take effect")

            async def healthy(phase: str) -> None:
                row = await asyncio.to_thread(survivor.request)
                state = snapshot(registry)
                evidence.emit("request", phase=phase, **row, state=state)
                current = state["models"][SURVIVOR]
                if current["pid"] != original_pid or not current["loaded"] or current["exit_code"] is not None:
                    raise RuntimeError("healthy pinned model changed process or residency")

            for _ in range(args.phase_requests):
                await healthy("before")
                await asyncio.sleep(args.interval_s)

            process = registry._loaded[VICTIM].adapter._process
            owned = psutil.Process(process.pid)
            if process.pid != original_victim_pid or owned.ppid() != os.getpid() or process.poll() is not None:
                raise RuntimeError("refusing to kill anything except the exact live child owned by this probe")
            evidence.emit(
                "crash",
                victim_pid=process.pid,
                survivor_pid=original_pid,
                child_parent_pid=owned.ppid(),
                command=owned.cmdline(),
                state=snapshot(registry),
            )
            process.kill()  # Exactly one deliberate signal, to this one owned process.
            await asyncio.to_thread(process.wait, timeout=10)

            async def recover() -> None:
                deadline = time.monotonic() + args.recovery_timeout_s
                while registry.get_failure(VICTIM) is None:
                    if time.monotonic() >= deadline:
                        raise RuntimeError("normal registry monitor did not detect the crashed engine")
                    await asyncio.sleep(0.1)
                evidence.emit("crash_detected", state=snapshot(registry))
                original_failure = registry.get_failure(VICTIM)
                while True:
                    failure = registry.get_failure(VICTIM)
                    if failure is not None and (failure != original_failure or failure.is_permanent):
                        raise RuntimeError("aborting before a second recovery load attempt")
                    row = await asyncio.to_thread(victim.request)
                    evidence.emit("recovery_request", **row, state=snapshot(registry))
                    if row["error"] is None:
                        new_pid = registry._loaded[VICTIM].adapter._process.pid
                        if new_pid == original_victim_pid:
                            raise RuntimeError("recovery reused the killed process identity")
                        remember_children(new_pid)
                        evidence.emit(
                            "recovered", old_pid=original_victim_pid, new_pid=new_pid, state=snapshot(registry)
                        )
                        return
                    statuses = [response["status"] for response in row["responses"]]
                    if (
                        statuses
                        and statuses[-1] == HTTPStatus.BAD_GATEWAY
                        and row["error"]["code"] == "MODEL_LOAD_FAILED"
                    ):
                        failure = registry.get_failure(VICTIM)
                        if failure is None or failure != original_failure or failure.cooldown_s is None:
                            raise RuntimeError("unexpected failed-model state")
                        delay = max(0.2, failure.cooldown_s - (time.monotonic() - failure.last_attempt_ts) + 0.1)
                        evidence.emit(
                            "waiting_before_fresh_request",
                            delay_s=delay,
                            reason="SDK does not retry MODEL_LOAD_FAILED; a later fresh request is required",
                        )
                    elif statuses and statuses[-1] == HTTPStatus.SERVICE_UNAVAILABLE:
                        retry = row["responses"][-1]["retry_after"]
                        delay = max(0.2, min(float(retry or 1), 30))
                    else:
                        raise RuntimeError("unexpected recovery response; stop instead of triggering more work")
                    if time.monotonic() >= deadline:
                        raise RuntimeError("recovery deadline exceeded")
                    await asyncio.sleep(delay)

            recovery_task = asyncio.create_task(recover())
            for _ in range(args.during_requests):
                await healthy("during")
                if recovery_task.done():
                    break
                await asyncio.sleep(args.interval_s)
            await recovery_task
            for _ in range(args.phase_requests):
                await healthy("after")
                await asyncio.sleep(args.interval_s)
            if any(value["errors"] for value in summarize(evidence.rows).values()):
                raise RuntimeError("at least one healthy-model request failed")
            passed = True
            evidence.emit("protocol_pass", survivor_pid=original_pid, state=snapshot(registry))
    except Exception as exc:
        evidence.emit("protocol_failed", error_type=type(exc).__name__, message=str(exc))
        logger.exception("recovery probe failed")
    finally:
        if recovery_task is not None and not recovery_task.done():
            recovery_task.cancel()
            await asyncio.gather(recovery_task, return_exceptions=True)
        for client in clients:
            client.close()
        inspection_errors: list[dict[str, Any]] = []
        try:
            children = psutil.Process(os.getpid()).children(recursive=True)
        except psutil.Error as exc:
            children = []
            inspection_errors.append({"stage": "enumerate", "error_type": type(exc).__name__})
        for child in children:
            try:
                child_identities.add((child.pid, child.create_time()))
            except psutil.NoSuchProcess:
                pass
            except psutil.Error as exc:
                inspection_errors.append({"stage": "identity", "pid": child.pid, "error_type": type(exc).__name__})
        evidence.emit(
            "before_shutdown",
            tracked_child_pids=sorted(pid for pid, _ in child_identities),
            inspection_errors=list(inspection_errors),
        )
        server.should_exit = True
        alive: list[int] = []
        try:
            await asyncio.wait_for(asyncio.shield(serve_task), timeout=90)
            for pid, created in child_identities:
                try:
                    child = psutil.Process(pid)
                    if child.create_time() == created and child.is_running() and child.status() != psutil.STATUS_ZOMBIE:
                        alive.append(pid)
                except psutil.NoSuchProcess:
                    pass
                except psutil.Error as exc:
                    inspection_errors.append({"stage": "liveness", "pid": pid, "error_type": type(exc).__name__})
            cleanup_ok = not inspection_errors and not alive and (registry is None or not registry.loaded_model_names)
            evidence.emit(
                "cleanup",
                okay=cleanup_ok,
                owned_children_alive=alive,
                inspection_errors=list(inspection_errors),
                state=snapshot(registry) if registry is not None else None,
            )
        except Exception as exc:  # noqa: BLE001 - Keep cleanup failure evidence even when shutdown raises.
            cleanup_ok = False
            evidence.emit(
                "cleanup_failed",
                error_type=type(exc).__name__,
                message=str(exc),
                owned_children_alive=alive,
                inspection_errors=list(inspection_errors),
            )
        sock.close()
        evidence.save(
            {
                "passed": passed and cleanup_ok,
                "protocol_passed": passed,
                "cleanup_ok": cleanup_ok,
                "smoke": args.smoke,
                "healthy_requests": summarize(evidence.rows),
                "crash_count": sum(row["event"] == "crash" for row in evidence.rows),
                "successful_reloads": sum(row["event"] == "recovered" for row in evidence.rows),
                "elapsed_s": time.monotonic() - evidence.started,
            }
        )
    return passed and cleanup_ok


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, required=True, help="New directory; existing evidence is never overwritten"
    )
    parser.add_argument("--smoke", action="store_true", help="Synthetic CPU adapters, real app/SDK/registry/processes")
    parser.add_argument("--phase-requests", type=int, default=25)
    parser.add_argument("--during-requests", type=int, default=50)
    parser.add_argument("--interval-s", type=float, default=2)
    parser.add_argument("--deadline-s", type=float, default=1300)
    parser.add_argument("--recovery-timeout-s", type=float, default=300)
    args = parser.parse_args()
    if args.phase_requests < 1 or args.during_requests < 1 or args.interval_s < 0:
        parser.error("request counts must be positive and interval nonnegative")
    if args.deadline_s <= 0 or args.deadline_s > MAX_DEADLINE_S or args.recovery_timeout_s <= 0:
        parser.error("deadline must be in (0, 1300] and recovery timeout positive")
    logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    evidence = Evidence(args.output)
    with ExitStack() as stack:
        if args.smoke:
            from smoke import install_smoke  # noqa: PLC0415 - Optional synthetic adapter, CPU smoke only.

            install_smoke(stack)
        try:
            okay = asyncio.run(run(args, evidence))
        except Exception as exc:
            logger.exception("probe setup failed")
            evidence.emit("setup_failed", error_type=type(exc).__name__, message=str(exc))
            evidence.save(
                {
                    "passed": False,
                    "smoke": args.smoke,
                    "setup_failed": True,
                    "healthy_requests": summarize(evidence.rows),
                }
            )
            okay = False
    raise SystemExit(0 if okay else 1)


if __name__ == "__main__":
    main()
