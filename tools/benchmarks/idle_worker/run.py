"""Compare idle-worker HTTP latency and throughput on one GPU via SIEClient."""

import argparse
import hashlib
import importlib.metadata
import itertools
import json
import math
import os
import platform
import shutil
import socket
import statistics
import subprocess
import sys
import tarfile
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from http import HTTPStatus
from pathlib import Path

import httpx
from sie_sdk import SIEClient
from sie_sdk.types import Item

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
LABELS = ["billing", "technical support", "account access", "sales", "other"]
TEXTS = [
    "The payment for my annual subscription was charged twice yesterday. Please check the invoice and refund the duplicate charge. I need the account to stay active while your team reviews the transaction.",
    "Our data export has stopped working after the latest update. The dashboard shows the job as complete, but the downloaded file contains only the column headings. We tried again from another browser with the same result. Please help us restore the missing records before the report is due tomorrow.",
    "I cannot log in to my account because the verification message never arrives. I checked the spam folder and confirmed that the email address is correct. My colleagues can still access their accounts. Please let me know how to recover access without creating a second account.",
    "We are evaluating the service for a team of twenty analysts and would like information about the available subscription plans. We expect to process several thousand documents each week. Could you explain how usage is measured and whether the service supports a shared workspace for our team? We would also like to schedule a short demonstration next week.",
]


def distribution(values: list[float]) -> dict:
    """Summarize samples using the nearest-rank definition of p95."""
    ordered = sorted(values)
    return {
        "n": len(values),
        "median": statistics.median(values),
        "p95": ordered[math.ceil(len(values) * 0.95) - 1],
        "max": max(values),
    }


def timing_sample(response: httpx.Response) -> dict[str, float] | None:
    if not response.request.url.path.startswith("/v1/extract/") or response.status_code != HTTPStatus.OK:
        return None
    return {
        key: float(response.headers[header])
        for key, header in (
            ("queue_ms", "X-Queue-Time"),
            ("inference_ms", "X-Inference-Time"),
            ("total_ms", "X-Total-Time"),
        )
    }


def measure(
    client: SIEClient, model: str, phase: str, concurrency: int, *, count: int = 0, duration: float = 0
) -> dict:
    barrier = threading.Barrier(concurrency)
    counter = itertools.count()
    start = 0.0
    latencies: list[float] = []

    def lane() -> None:
        barrier.wait()
        while True:
            index = next(counter)
            if (count and index >= count) or (duration and time.perf_counter() - start >= duration):
                return
            before = time.perf_counter()
            result = client.extract(
                model,
                Item(text=TEXTS[index % len(TEXTS)], id=f"{phase}:{index}"),
                labels=LABELS,
                wait_for_capacity=False,
                max_oom_retries=0,
            )
            if result.get("error"):
                raise RuntimeError(result["error"])
            if not result.get("classifications"):
                raise RuntimeError("No classifications returned")
            latencies.append((time.perf_counter() - before) * 1000)

    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        start = time.perf_counter()
        futures = [executor.submit(lane) for _ in range(concurrency)]
        for future in futures:
            future.result()
    elapsed = time.perf_counter() - start
    return {
        "phase": phase,
        "concurrency": concurrency,
        "elapsed_s": elapsed,
        "requests": len(latencies),
        "requests_per_s": len(latencies) / elapsed,
        "latency_ms": distribution(latencies),
    }


def wait_for_server(process: subprocess.Popen, port: int) -> None:
    deadline = time.monotonic() + 300
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"Server exited with {process.returncode}")
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=1):
                return
        except OSError:
            time.sleep(0.25)
    raise TimeoutError("Server did not bind within 300 seconds")


def trial(args: argparse.Namespace, arm: str, trial_index: int, source: Path, output: Path) -> dict:
    name = f"{trial_index:02d}-{arm}"
    samples: list[dict] = []

    def capture(response: httpx.Response) -> None:
        if (sample := timing_sample(response)) is not None:
            samples.append(sample)

    environment = os.environ.copy()
    environment.update(
        {
            "PYTHONPATH": str(source / "packages/sie_server/src") + os.pathsep + environment.get("PYTHONPATH", ""),
            "SIE_BUNDLE": "default",
            "SIE_TELEMETRY_DISABLED": "true",
            "SIE_TORCH_NUM_THREADS": "4",
            "TOKENIZERS_PARALLELISM": "false",
            "PYTHONUNBUFFERED": "1",
            "SIE_LOG_LEVEL": "warning",
        }
    )
    trace = output / f"{name}-batches.jsonl"
    argv = [
        sys.executable,
        str(HERE / "server.py"),
        "--model",
        args.model,
        "--models-dir",
        str(source / "packages/sie_server/models"),
        "--trace",
        str(trace),
        "--port",
        str(args.port),
        "--device",
        args.device,
    ]
    phases = []
    result = {"arm": arm, "trial": trial_index, "phases": phases, "status": "running"}
    result_path = output / f"{name}.json"

    def persist_result() -> None:
        pending = result_path.with_suffix(".json.tmp")
        pending.write_text(json.dumps(result, indent=2) + "\n")
        pending.replace(result_path)

    def record_phase(phase: dict) -> None:
        phases.append(phase)
        persist_result()

    with (output / f"{name}-server.log").open("w") as log:
        process = subprocess.Popen(argv, env=environment, stdout=log, stderr=subprocess.STDOUT)  # noqa: S603 — fixed local server, no shell.
        try:
            wait_for_server(process, args.port)
            base_url = f"http://127.0.0.1:{args.port}"
            transport = httpx.Client(
                base_url=base_url,
                timeout=300,
                event_hooks={"response": [capture]},
                limits=httpx.Limits(max_connections=64, max_keepalive_connections=64),
            )
            with SIEClient(base_url, timeout_s=300, http_client=transport) as client:
                for concurrency in (1, 8, 32):
                    measure(client, args.model, f"warmup{concurrency}", concurrency, count=64)
                samples.clear()
                sequential = measure(client, args.model, "sequential", 1, count=args.sequential)
                if len(samples) != args.sequential:
                    raise RuntimeError(f"Expected {args.sequential} successful timing samples, got {len(samples)}")
                sequential["queue_ms"] = distribution([sample["queue_ms"] for sample in samples])
                sequential["inference_ms"] = distribution([sample["inference_ms"] for sample in samples])
                sequential["queue_under_1ms"] = sum(sample["queue_ms"] < 1 for sample in samples)
                sequential["samples"] = samples.copy()
                record_phase(sequential)
                for concurrency in (8, 32):
                    for burst in range(4):
                        time.sleep(0.05)
                        record_phase(
                            measure(client, args.model, f"burst{concurrency}-{burst}", concurrency, count=concurrency)
                        )
                    measure(client, args.model, f"warmthroughput{concurrency}", concurrency, count=128)
                    record_phase(
                        measure(client, args.model, f"throughput{concurrency}", concurrency, duration=args.duration)
                    )
        except BaseException as error:
            result["status"] = "failed"
            try:
                persist_result()
            except OSError as persistence_error:
                error.add_note(f"Could not persist failed trial: {persistence_error}")
            raise
        finally:
            process.terminate()
            try:
                process.wait(timeout=40)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
    result.update(status="complete", server_returncode=process.returncode)
    if trace.exists():
        batches = [json.loads(line) for line in trace.read_text().splitlines()]
        for phase in phases:
            prefix = phase["phase"] + ":"
            phase["worker_batch_sizes"] = [
                record["size"] for record in batches if any(i.startswith(prefix) for i in record["request_ids"])
            ]
    else:
        result["status"] = "timing_complete_trace_missing"
    persist_result()
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="knowledgator/gliclass-large-v1.0")
    parser.add_argument("--output", type=Path, default=HERE / "results")
    parser.add_argument("--duration", type=float, default=12)
    parser.add_argument("--sequential", type=int, default=100)
    parser.add_argument("--order", default="baseline,candidate,candidate,baseline,baseline,candidate")
    parser.add_argument("--port", type=int, default=8373)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.duration <= 0 or args.sequential < 1:
        parser.error("duration and sequential must be positive")
    order = args.order.split(",")
    if not order or any(arm not in {"baseline", "candidate"} for arm in order):
        parser.error("order must contain baseline/candidate arms")
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((HERE / "_inputs/manifest.json").read_text())
    nvidia_smi = shutil.which("nvidia-smi")
    if nvidia_smi is None:
        raise RuntimeError("nvidia-smi is required")
    environment = {
        "python": platform.python_version(),
        "packages": {
            name: importlib.metadata.version(name)
            for name in ("torch", "transformers", "gliclass", "gliformer", "httpx", "uvicorn")
        },
        "nvidia_smi": subprocess.check_output(  # noqa: S603 — fixed GPU metadata query, no shell.
            [nvidia_smi, "--query-gpu=name,uuid,driver_version,memory.total", "--format=csv,noheader"], text=True
        ).strip(),
        "cpu_count": os.cpu_count(),
        "torch_threads": 4,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
    }
    result = {
        "manifest": manifest,
        "environment": environment,
        "model": args.model,
        "texts": TEXTS,
        "labels": LABELS,
        "trials": [],
    }
    with tempfile.TemporaryDirectory(prefix="sie-idle-worker-") as temporary:
        roots = {}
        for arm in set(order):
            archive = HERE / f"_inputs/{arm}.tar"
            if hashlib.sha256(archive.read_bytes()).hexdigest() != manifest[arm]["archive_sha256"]:
                raise RuntimeError(f"{arm} archive hash mismatch")
            roots[arm] = Path(temporary) / arm
            roots[arm].mkdir()
            with tarfile.open(archive) as snapshot:
                snapshot.extractall(roots[arm], filter="data")
        for index, arm in enumerate(order):
            print(f"IDLE_WORKER_START trial={index} arm={arm}", flush=True)
            current = trial(args, arm, index, roots[arm], args.output)
            result["trials"].append(current)
            (args.output / "results.json").write_text(json.dumps(result, indent=2) + "\n")
            print(
                json.dumps(
                    {
                        "arm": arm,
                        "trial": index,
                        "phases": [
                            {k: v for k, v in p.items() if k not in {"samples", "worker_batch_sizes"}}
                            for p in current["phases"]
                        ],
                    }
                ),
                flush=True,
            )
    print("IDLE_WORKER_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
