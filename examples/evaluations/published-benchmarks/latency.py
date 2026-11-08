"""Time one request at a time against your own server, beside the per-request times the SIE website shows."""

import contextlib
import hashlib
import json
import statistics
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from sie_sdk import ModelLoadingError, ProvisioningError, SIEConnectionError

from official import Fetch
from score import write_json

DATASET = "superlinked/sie-task-evidence"
RESOLVE_URL = f"https://huggingface.co/datasets/{DATASET}/resolve"
STUDY_REVISION = "2f74a5c09c3cefa0f5f754f92949600fcd4dc1aa"
STUDY = "latency-fastest-config/2026-10-11"
JOBS_SHA256 = f"{STUDY}/prereg/jobs-sha256.json"
LEVEL = "c1"
WARMUP_REQUESTS = 20
WARMUP_TAG_BASE = 10_000_000
CLASSES = ("l4", "rtx", "h100")
REQUEST_TIMEOUT_S = 300.0
READY_TIMEOUT_S = 900.0
READY_POLL_S = 5.0
NOT_READY = (SIEConnectionError, ModelLoadingError, ProvisioningError)
NS_PER_MS = 1_000_000


class InputMismatch(Exception):
    pass


@dataclass(frozen=True)
class Page:
    name: str
    input_revision: str
    input_path: str
    build: Callable[[bytes], dict[str, Any]]


def classify_jobs(raw: bytes) -> dict[str, Any]:
    items = [json.loads(line) for line in raw.decode("utf-8").splitlines() if line.strip()]
    jobs = [
        {"id": it["id"], "text": it["text"], "labels": list(it["labels"]), "meta": {"domain": it["domain"]}}
        for it in items
    ]
    return {
        "page": "classify",
        "model": "fastino/GLiNER2.5-Decide",
        "profile": "default",
        "op": "extract",
        "defaults": {"options": {"overflow_policy": "error"}},
        "source": {
            "items": "classify-current-arms-2026-10-11/prereg/items.jsonl @4e9f6498",
            "items_sha256": hashlib.sha256(raw).hexdigest(),
            "request": "one message per extract call with its 8 labels, options {overflow_policy: error}",
        },
        "jobs": jobs,
    }


def search_jobs(raw: bytes) -> dict[str, Any]:
    questions = json.loads(raw.decode("utf-8"))["questions"]
    return {
        "page": "search",
        "model": "Qwen/Qwen3-Embedding-4B",
        "profile": "default",
        "op": "encode",
        "defaults": {"output_types": ["dense"], "is_query": True},
        "warm_variant": "tagged",
        "source": {
            "questions": "lookalike-search/inputs/questions.json @b939e410 (440 questions)",
            "sha256": hashlib.sha256(raw).hexdigest(),
            "request": "one question per encode call, is_query true, output_types [dense], no instruction sent (server default)",
        },
        "jobs": [{"id": q["id"], "text": q["question"]} for q in questions],
    }


PAGES = {
    "classify": Page(
        "classify",
        "4e9f649842dd802d353d10fc8dc2347b9a927189",
        "classify-current-arms-2026-10-11/prereg/items.jsonl",
        classify_jobs,
    ),
    "search": Page(
        "search",
        "b939e41046eef6a0c2a9ee60c152161fd3fb7e04",
        "lookalike-search/inputs/questions.json",
        search_jobs,
    ),
}


def dataset_url(revision: str, path: str) -> str:
    return f"{RESOLVE_URL}/{revision}/{path}"


def jobs_bytes(doc: Mapping[str, Any]) -> bytes:
    return json.dumps(doc).encode("utf-8")


def load_jobs(page: Page, fetch: Fetch) -> tuple[dict[str, Any], str]:
    expected = json.loads(fetch(dataset_url(STUDY_REVISION, JOBS_SHA256), None))[f"{page.name}.json"]["sha256"]
    doc = page.build(fetch(dataset_url(page.input_revision, page.input_path), None))
    digest = hashlib.sha256(jobs_bytes(doc)).hexdigest()
    if digest != expected:
        raise InputMismatch(f"the rebuilt {page.name} jobs have sha256 {digest}, the published jobs file {expected}")
    return doc, digest


def load_published(page: Page, fetch: Fetch) -> dict[str, Any]:
    doc = json.loads(fetch(dataset_url(STUDY_REVISION, f"{STUDY}/results/{page.name}.json"), None))
    classes = {}
    for key in CLASSES:
        entry = doc["classes"][key]
        latency = entry["levels"][LEVEL]["latency"]
        classes[key] = {
            "gpu": entry["gpu"],
            "n": latency["n"],
            "p50_ms": latency["p50_ms"],
            "p95_ms": latency["p95_ms"],
            "mean_ms": latency["mean_ms"],
            "bundle": entry["bundle"],
            "image_digest": entry["image_digest"],
            "server_overlay_revision": entry["server_overlay_revision"],
            "client_placement": entry["client_placement"],
        }
    return {
        "dataset": DATASET,
        "revision": STUDY_REVISION,
        "path": f"{STUDY}/results/{page.name}.json",
        "level": LEVEL,
        "best_class": doc["best_class"]["key"],
        "classes": classes,
    }


def percentile(values: Sequence[float], q: float) -> float:
    ordered = sorted(values)
    k = (len(ordered) - 1) * q
    low = int(k)
    high = min(low + 1, len(ordered) - 1)
    return ordered[low] + (ordered[high] - ordered[low]) * (k - low)


def latency_stats(values_ms: Sequence[float]) -> dict[str, Any]:
    if not values_ms:
        return {"n": 0, "p50_ms": None, "p95_ms": None, "mean_ms": None}
    return {
        "n": len(values_ms),
        "p50_ms": round(percentile(values_ms, 0.5), 2),
        "p95_ms": round(percentile(values_ms, 0.95), 2),
        "mean_ms": round(statistics.fmean(values_ms), 2),
    }


def warmup_jobs(doc: Mapping[str, Any]) -> list[dict[str, Any]]:
    jobs = doc["jobs"]
    tagged = doc.get("warm_variant") == "tagged"
    warm = []
    for i in range(WARMUP_REQUESTS):
        job = dict(jobs[-1 - (i % len(jobs))])
        if tagged:
            job["text"] = f"[{LEVEL}-{WARMUP_TAG_BASE + i}] " + job["text"]
        warm.append(job)
    return warm


def send(client: Any, doc: Mapping[str, Any], job: Mapping[str, Any], *, wait_for_capacity: bool) -> Any:
    item = {"id": job["id"], "text": job["text"]}
    defaults = doc.get("defaults") or {}
    if doc["op"] == "extract":
        return client.extract(
            doc["model"],
            item,
            labels=list(job["labels"]),
            options=dict(defaults["options"]),
            wait_for_capacity=wait_for_capacity,
            provision_timeout_s=READY_TIMEOUT_S if wait_for_capacity else REQUEST_TIMEOUT_S,
            max_oom_retries=0,
        )
    return client.encode(
        doc["model"],
        item,
        output_types=list(defaults["output_types"]),
        is_query=bool(defaults["is_query"]),
        wait_for_capacity=wait_for_capacity,
        provision_timeout_s=READY_TIMEOUT_S if wait_for_capacity else REQUEST_TIMEOUT_S,
        max_oom_retries=0,
    )


def wait_until_ready(
    client: Any,
    doc: Mapping[str, Any],
    job: Mapping[str, Any],
    *,
    monotonic: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
) -> None:
    deadline = monotonic() + READY_TIMEOUT_S
    while True:
        try:
            send(client, doc, job, wait_for_capacity=True)
            return
        except NOT_READY:
            if monotonic() >= deadline:
                raise
            sleep(READY_POLL_S)


def warm_up(client: Any, doc: Mapping[str, Any], **ready: Any) -> None:
    jobs = warmup_jobs(doc)
    wait_until_ready(client, doc, jobs[0], **ready)
    for job in jobs[1:]:
        with contextlib.suppress(Exception):
            send(client, doc, job, wait_for_capacity=False)


def timed_requests(client: Any, doc: Mapping[str, Any], clock: Callable[[], int] = time.perf_counter_ns) -> list[dict]:
    rows = []
    for job in doc["jobs"]:
        start = clock()
        try:
            send(client, doc, job, wait_for_capacity=False)
        except Exception as exc:  # noqa: BLE001
            status = f"error:{type(exc).__name__}"
        else:
            status = "retried" if getattr(client, "last_retry_count", 0) else "ok"
        end = clock()
        rows.append({"id": job["id"], "status": status, "latency_ms": round((end - start) / NS_PER_MS, 3)})
    return rows


def differences(published: Mapping[str, Any]) -> list[str]:
    best = published["classes"][published["best_class"]]
    return [
        (
            "the published client ran in the server's container and sent each request over the sidecar's local "
            "Unix socket, with no HTTP; this run sends HTTP through sie_sdk.SIEClient, so its time also includes "
            "the HTTP server and your network path"
        ),
        (
            "this run uses your GPU and the server you started; the published run used the "
            f"{best['bundle']} image {best['image_digest']} with the server package from public main "
            f"{best['server_overlay_revision'][:8]}"
        ),
    ]


def _timestamp(moment: datetime) -> str:
    return moment.strftime("%Y-%m-%dT%H:%M:%SZ")


def build_record(
    page: Page,
    doc: Mapping[str, Any],
    digest: str,
    published: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
    *,
    endpoint: str,
    client_description: str,
    started_at: datetime,
    finished_at: datetime,
) -> dict[str, Any]:
    ok = [row["latency_ms"] for row in rows if row["status"] == "ok"]
    stats = latency_stats(ok)
    return {
        "kind": "published-benchmark-reproduction",
        "schema_version": 1,
        "evaluation_mode": "latency",
        "page": page.name,
        "model": doc["model"],
        "operation": doc["op"],
        "ingress": {"url": endpoint},
        "inputs": {
            "dataset": DATASET,
            "revisions": {"study": STUDY_REVISION, "inputs": page.input_revision},
            "path": page.input_path,
            "jobs_sha256": digest,
            "jobs": len(doc["jobs"]),
        },
        "warmup_requests": WARMUP_REQUESTS,
        "client": {"description": client_description, "worker_count": 1, "processes_per_worker": 1},
        "run": {
            "started_at": _timestamp(started_at),
            "finished_at": _timestamp(finished_at),
            "endpoint": endpoint,
            "request_count": len(rows),
            "failure_count": len(rows) - len(ok),
        },
        "metrics": [
            {"name": "latency_p50_ms", "unit": "ms", "aggregation": "p50", "value": stats["p50_ms"]},
            {"name": "latency_p95_ms", "unit": "ms", "aggregation": "p95", "value": stats["p95_ms"]},
            {"name": "latency_mean_ms", "unit": "ms", "aggregation": "mean", "value": stats["mean_ms"]},
        ],
        "published": published,
        "differences": differences(published),
        "requests": list(rows),
    }


def _cell(value: float | None) -> str:
    return "-" if value is None else f"{value:.2f}"


def format_report(record: Mapping[str, Any]) -> list[str]:
    published = record["published"]
    metrics = {metric["name"]: metric["value"] for metric in record["metrics"]}
    n = record["run"]["request_count"] - record["run"]["failure_count"]
    rows = [("yours", n, metrics["latency_p50_ms"], metrics["latency_p95_ms"], metrics["latency_mean_ms"])]
    for entry in published["classes"].values():
        rows.append((f"published {entry['gpu']}", entry["n"], entry["p50_ms"], entry["p95_ms"], entry["mean_ms"]))
    width = max(len(row[0]) for row in rows)
    lines = [
        (
            f"{record['page']}: {record['model']}, {record['operation']}, one request at a time, "
            f"{record['inputs']['jobs']} inputs after {record['warmup_requests']} warm-up requests"
        ),
        f"{'':{width}}  {'n':>5}  {'p50 ms':>8}  {'p95 ms':>8}  {'mean ms':>8}",
    ]
    lines += [
        f"{name:{width}}  {count:>5}  {_cell(p50):>8}  {_cell(p95):>8}  {_cell(mean):>8}"
        for name, count, p50, p95, mean in rows
    ]
    best = published["classes"][published["best_class"]]
    lines.append(f"the website shows the published {best['gpu']} p50, rounded up")
    lines += [f"differs: {line}" for line in record["differences"]]
    failed = record["run"]["failure_count"]
    if failed:
        statuses = sorted({row["status"] for row in record["requests"] if row["status"] != "ok"})
        lines.append(
            f"failed: {failed} of {record['run']['request_count']} ({', '.join(statuses)}), left out of the numbers"
        )
    return lines


def run_latency(
    page_name: str,
    client: Any,
    fetch: Fetch,
    *,
    endpoint: str,
    client_description: str,
    results_dir: Path,
    clock: Callable[[], int] = time.perf_counter_ns,
    **ready: Any,
) -> tuple[dict[str, Any], Path]:
    page = PAGES[page_name]
    doc, digest = load_jobs(page, fetch)
    published = load_published(page, fetch)
    started_at = datetime.now(UTC)
    warm_up(client, doc, **ready)
    rows = timed_requests(client, doc, clock)
    finished_at = datetime.now(UTC)
    record = build_record(
        page,
        doc,
        digest,
        published,
        rows,
        endpoint=endpoint,
        client_description=client_description,
        started_at=started_at,
        finished_at=finished_at,
    )
    results_dir.mkdir(parents=True, exist_ok=True)
    path = results_dir / f"latency-{page.name}-{started_at.strftime('%Y%m%dT%H%M%SZ')}.json"
    write_json(path, record)
    return record, path
