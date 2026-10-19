"""Record fixed-length, serial SDK chat streams with bounded child lifetimes."""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import math
import signal
import subprocess
import sys
import time
from collections.abc import Iterator
from pathlib import Path
from types import GeneratorType
from typing import Any
from urllib.parse import urlsplit

import httpx
from sie_sdk import SIEClient

from tools.benchmarks.stream_decode.analysis import require
from tools.benchmarks.stream_decode.observer import MAX_EVENTS, SOURCE_FILES, VERSION, clock_domain, digest, save

MIN_TOKENS, MAX_TOKENS = 2, 4096
MAX_TRIALS, MAX_REQUEST_SECONDS = 100, 600
CHILD_CLEANUP_SECONDS = 6


@contextlib.contextmanager
def child_deadline(seconds: float) -> Iterator[None]:
    """A wall deadline survives parent death and a stream of tiny HTTP reads."""

    def expired(_signum: int, _frame: Any) -> None:
        raise TimeoutError("fixed child wall deadline expired")

    previous = signal.signal(signal.SIGALRM, expired)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)


def one(plan: dict[str, Any], index: int, output: Path) -> None:
    """A fresh process owns one stream, its partial record, and no retry."""
    record: dict[str, Any] = {
        "index": index,
        "warmup": index < plan["warmups"],
        "completed": False,
        "clock_domain": clock_domain(),
        "chunks": [],
        "http_requests": 0,
        "plan_sha256": digest(plan),
        "cohort": plan["cohort"],
    }

    def request(req: httpx.Request) -> None:
        record["http_requests"] += 1
        require(record["http_requests"] == 1, "benchmark refuses a second HTTP attempt")
        require(req.url.path == "/v1/chat/completions", "unexpected SDK endpoint")
        record["request_body"] = json.loads(req.content)

    def interrupted(_signum: int, _frame: Any) -> None:
        raise TimeoutError("fixed per-request deadline expired")

    previous = signal.signal(signal.SIGTERM, interrupted)
    record["started_ns"] = time.perf_counter_ns()
    try:
        with (
            child_deadline(plan["request_seconds"]),
            httpx.Client(base_url=plan["url"], trust_env=False, event_hooks={"request": [request]}) as http,
            SIEClient(plan["url"], remote="forbid", timeout_s=plan["request_seconds"], http_client=http) as client,
        ):
            stream = client.stream_chat_completions(
                plan["model"],
                [{"role": "user", "content": plan["prompt"]}],
                max_completion_tokens=plan["completion_tokens"],
                temperature=0,
                top_p=1,
                top_k=1,
                presence_penalty=0,
                frequency_penalty=0,
                repetition_penalty=1,
                seed=17,
                stream_options={"include_usage": True},
                include_reasoning=False,
                extra_body={
                    "min_tokens": plan["completion_tokens"],
                    "chat_template_kwargs": {"enable_thinking": False},
                },
                wait_for_capacity=False,
                max_oom_retries=0,
                provision_timeout_s=plan["request_seconds"],
            )
            if not isinstance(stream, GeneratorType):
                raise TypeError("the SDK stream must support deterministic generator closure")
            # Client construction is outside the timed request.
            record["started_ns"] = time.perf_counter_ns()
            try:
                with contextlib.closing(stream):
                    for chunk in stream:
                        at_ns = time.perf_counter_ns()
                        record["chunks"].append({"at_ns": at_ns, "chunk": chunk})
                        require(len(record["chunks"]) <= MAX_EVENTS, "SDK stream exceeded fixed event bound")
                record["finished_ns"] = time.perf_counter_ns()
                record["completed"] = True
            finally:
                record["retry_count"] = client.last_retry_count
    except BaseException as error:
        record["error"] = {"type": type(error).__name__, "message": str(error)}
        raise
    finally:
        record.setdefault("finished_ns", time.perf_counter_ns())
        signal.signal(signal.SIGTERM, previous)
        save(output, record)


def stop_child(process: subprocess.Popen[Any]) -> list[str]:
    signals = []
    if process.poll() is None:
        with contextlib.suppress(ProcessLookupError):
            process.terminate()
        signals.append("SIGTERM")
        try:
            process.wait(timeout=3)
        except subprocess.TimeoutExpired:
            with contextlib.suppress(ProcessLookupError):
                process.kill()
            signals.append("SIGKILL")
            process.wait(timeout=3)
    return signals


def run(args: argparse.Namespace) -> None:
    target = urlsplit(args.url)
    require(
        target.scheme == "http" and target.hostname in {"127.0.0.1", "localhost", "::1"},
        "benchmark client requires its dedicated local HTTP server",
    )
    require(
        target.username is None and target.password is None and not target.query and not target.fragment,
        "URL must not contain credentials, query, or fragment",
    )
    require(MIN_TOKENS <= args.completion_tokens <= MAX_TOKENS, "completion token count must be between 2 and 4096")
    require(0 <= args.warmups <= MAX_TRIALS and 1 <= args.repeats <= MAX_TRIALS, "invalid fixed trial count")
    require(
        math.isfinite(args.total_seconds)
        and 0 < args.request_seconds <= MAX_REQUEST_SECONDS
        and args.total_seconds > args.request_seconds + CHILD_CLEANUP_SECONDS,
        "invalid child/total deadline",
    )
    require(not args.output.exists(), "use a fresh output directory")
    prompt = args.prompt.read_text()
    require(bool(prompt.strip()), "prompt is empty")
    manifest = json.loads(args.server_manifest.read_text())
    markers = [json.loads(path.read_text()) for path in args.server_manifest.parent.glob("installed-*.json")]
    require(
        any(
            marker.get("version") == VERSION
            and marker.get("source_files") == SOURCE_FILES
            and marker.get("server_manifest_sha256") == digest(manifest)
            for marker in markers
        ),
        "matching child observer is not installed; preload and verify readiness first",
    )
    plan = {
        "schema": 2,
        "url": args.url,
        "model": args.model,
        "prompt": prompt,
        "cohort": manifest["cohort"],
        "server_manifest_sha256": digest(manifest),
        "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
        "completion_tokens": args.completion_tokens,
        "warmups": args.warmups,
        "repeats": args.repeats,
        "request_seconds": args.request_seconds,
        "total_seconds": args.total_seconds,
        "child_cleanup_seconds": CHILD_CLEANUP_SECONDS,
        "clock_domain": clock_domain(),
        "concurrency": 1,
        "sampling": {
            "temperature": 0,
            "top_p": 1,
            "top_k": 1,
            "presence_penalty": 0,
            "frequency_penalty": 0,
            "repetition_penalty": 1,
            "seed": 17,
            "min_tokens": args.completion_tokens,
            "max_completion_tokens": args.completion_tokens,
        },
    }
    args.output.mkdir(parents=True)
    save(args.output / "plan.json", plan)
    deadline = time.monotonic() + args.total_seconds
    summary: dict[str, Any] = {"completed": False, "children": []}
    try:
        for index in range(args.warmups + args.repeats):
            require(
                deadline - time.monotonic() > args.request_seconds + CHILD_CLEANUP_SECONDS,
                "insufficient time to admit fixed child",
            )
            destination = args.output / f"trial-{index:04d}.json"
            with (args.output / f"trial-{index:04d}.log").open("wb") as log:
                process = subprocess.Popen(  # noqa: S603 - fixed module, no shell
                    [
                        sys.executable,
                        "-m",
                        "tools.benchmarks.stream_decode.client",
                        "--one",
                        str(args.output / "plan.json"),
                        "--index",
                        str(index),
                        "--output",
                        str(destination),
                    ],
                    stdout=log,
                    stderr=subprocess.STDOUT,
                )
                child: dict[str, Any] = {"index": index, "pid": process.pid, "timed_out": False}
                summary["children"].append(child)
                try:
                    child["returncode"] = process.wait(timeout=args.request_seconds)
                except subprocess.TimeoutExpired:
                    child["timed_out"] = True
                finally:
                    child["signals"] = stop_child(process)
                    child["returncode"] = process.returncode
                    save(args.output / "controller.json", summary)
                require(not child["timed_out"] and child["returncode"] == 0, "stream child failed; raw record retained")
        summary["completed"] = True
    finally:
        save(args.output / "controller.json", summary)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8080")
    parser.add_argument("--model")
    parser.add_argument("--prompt", type=Path)
    parser.add_argument("--server-manifest", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--completion-tokens", type=int, default=128)
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=6)
    parser.add_argument("--request-seconds", type=float, default=120)
    parser.add_argument("--total-seconds", type=float, default=1200)
    parser.add_argument("--one", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--index", type=int, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.one:
        one(json.loads(args.one.read_text()), args.index, args.output)
    else:
        require(
            args.model is not None and args.prompt is not None and args.server_manifest is not None,
            "--model, --prompt and --server-manifest are required",
        )
        run(args)


if __name__ == "__main__":
    main()
