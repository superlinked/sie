"""Repeat the published benchmark numbers, or test a new model the same way, from one client process."""

import argparse
import contextlib
import json
import math
import os
import platform
import re
import shlex
import sys
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor, wait
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from fractions import Fraction
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit, urlunsplit

import httpx
import numpy as np
import sie_sdk
from scipy.sparse import csr_matrix
from sie_sdk import SIEClient

import latency as latency_mode
from official import (
    MODEL_ID,
    SKIPPED,
    Fetch,
    Lookup,
    LookupFailed,
    NotFound,
    checked_baseline_value,
    http_fetch,
    lookup_official,
    plain_text,
)
from score import (
    BASELINE_METRICS,
    HF_DATASETS,
    MAIN_SCORE,
    NS_PER_MS,
    NS_PER_S,
    PERFORMANCE_MODE,
    PRIMARY_METRIC,
    SCORER_ID,
    Qrels,
    Rows,
    compare,
    compute_verdict,
    eval_split,
    format_rows,
    format_verdict,
    load_corpus,
    load_qrels,
    load_queries,
    load_top_ranked,
    metric_rows,
    nanoseconds,
    parse_task_key,
    performance_metric_rows,
    performance_metrics,
    scorer_version,
    task_metrics,
    write_json,
    write_jsonl,
)

DEFAULT_ENDPOINT = "https://api.superlinked.com"
DEFAULT_TIMEOUT_S = 300.0
TOP_K = 1000
DOCUMENT_CHUNK = 500
QUERY_TOKEN_ROWS = 256
COSINE_NORM_FLOOR = 1e-12
RESULTS_DIR = Path("results")
SCORE_TASKS = frozenset(
    {
        "mteb/AskUbuntuDupQuestions",
        "mteb/MMarcoReranking",
        "mteb/T2Reranking",
        "mteb/CMedQAv1-reranking",
        "mteb/CMedQAv2-reranking",
    }
)
SCORE_ITEMS_PER_REQUEST = 100
CANDIDATE_ENCODE_ITEMS_PER_REQUEST = 64
QUALITY_PARAMETERS = frozenset({"limit", "candidates_model", "candidates_k"})
PERFORMANCE_PARAMETERS = frozenset({"limit", "candidates-k", "split", "filter", "role"})
ENCODE_OUTPUTS = ("sparse", "dense", "multivector")
MUVERA_PROFILE = "muvera"
REFERENCE_KIND = "published-benchmark-reference"
RESULT_KIND = "published-benchmark-reproduction"
PERFORMANCE_LIMITS = {"encode": (30.0, 64), "score": (300.0, 128)}
DEFAULT_DURATION_S = 60.0
DEFAULT_WARMUP_S = 30.0
DEFAULT_SETTLE_S = 10.0
DEFAULT_DRAIN_S = 10.0
DEFAULT_READINESS_TIMEOUT_S = 300.0
DEFAULT_LAG_TOLERANCE_MS = 20.0
RETRY_POLICY = {"backoff": "none", "max_attempts": 1, "record_attempts": True}
ROLES = ("queries", "corpus")
COMMAND_PREFIX = ("uv", "run", "python", "run.py")

Corpus = dict[str, dict[str, str]]
Queries = dict[str, str]
TopRanked = dict[str, list[str]]
PredictionMap = dict[str, dict[str, float]]
ScoreRequest = tuple[dict[str, str], list[dict[str, str]]]


class RequestFailed(Exception):
    pass


@dataclass(frozen=True)
class QualityPlan:
    reference_id: str | None
    reference_metrics: list[dict[str, Any]]
    model: str
    profile: str
    operation: str
    task_key: str
    task_name: str
    task_params: dict[str, str]
    revision: str
    output_contract: str
    items_per_request: int | None
    request_timeout_s: float
    endpoint: str
    model_version: str | None
    comparable_reasons: list[str]
    baseline: dict[str, Any] | None = None
    instruction: str | None = None
    candidates_model: str | None = None
    candidates_k: int | None = None

    @property
    def route(self) -> str:
        return self.model if self.profile == "default" else f"{self.model}:{self.profile}"


@dataclass(frozen=True)
class PerformancePlan:
    reference_id: str | None
    reference_metrics: list[dict[str, Any]]
    model: str
    profile: str
    operation: str
    task_key: str
    task_name: str
    task_params: dict[str, str]
    revision: str
    output_contract: str
    items_per_request: int | None
    rate: float
    duration_s: float
    warmup_s: float
    settle_s: float
    drain_s: float
    request_timeout_s: float
    target_readiness_timeout_s: float
    max_in_flight: int
    lag_tolerance_ms: float
    endpoint: str
    model_version: str | None
    comparable_reasons: list[str]

    @property
    def route(self) -> str:
        return self.model if self.profile == "default" else f"{self.model}:{self.profile}"

    @property
    def level_name(self) -> str:
        return load_level_name(self.rate)


@dataclass(frozen=True)
class Loaders:
    corpus: Callable[[str, str], Corpus] = load_corpus
    queries: Callable[[str, str], Queries] = load_queries
    qrels: Callable[[str, str], Qrels] = load_qrels
    top_ranked: Callable[[str, str], TopRanked] = load_top_ranked


@dataclass(frozen=True)
class QualityData:
    corpus: Corpus
    queries: Queries
    qrels: Qrels
    top_ranked: TopRanked | None


@dataclass(frozen=True)
class PerformanceData:
    selected_ids: list[str]
    payloads: list[dict[str, str]] | list[ScoreRequest]


@dataclass
class CallLog:
    request_count: int = 0
    failure_count: int = 0
    sdk_retries: int = 0
    execution_revisions: list[str] = field(default_factory=list)
    encoder_revisions: list[str] = field(default_factory=list)
    failure: str | None = None
    readiness_failure: str | None = None

    def record(self, client: Any) -> None:
        self.request_count += 1
        self.sdk_retries += int(client.last_retry_count)
        revision = client.last_model_revision
        if revision and revision not in self.execution_revisions:
            self.execution_revisions.append(revision)

    def fail(self, client: Any, exc: Exception) -> None:
        self.request_count += 1
        self.failure_count += 1
        self.sdk_retries += int(client.last_retry_count)
        self.failure = f"{type(exc).__name__}: {exc}"

    @property
    def revision_changed(self) -> bool:
        return len(self.execution_revisions) > 1 or len(self.encoder_revisions) > 1


def _add_shared_arguments(command: argparse.ArgumentParser) -> None:
    command.add_argument("--reference", type=Path, help="A references/<id>.json file")
    command.add_argument("--model", help="Model route, model[:profile]")
    command.add_argument("--task", help="Task key, for example mteb/SciFact")
    command.add_argument("--output", choices=ENCODE_OUTPUTS, help="Output contract of an encode task")
    command.add_argument("--items-per-request", type=int, help="Items per encode request")
    command.add_argument("--endpoint", default=DEFAULT_ENDPOINT, help="SIE endpoint URL")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    quality = commands.add_parser(
        "quality", help="Measure quality against a published record, an official number or a supplied baseline"
    )
    _add_shared_arguments(quality)
    quality.add_argument("--baseline", help="The number to compare with, <metric>=<value>, for example ndcg_at_10=0.32")
    quality.add_argument("--baseline-source", help="Where the baseline comes from: model card, paper or leaderboard")
    quality.add_argument("--instruction", help="Instruction sent with queries only, on encode tasks")
    performance = commands.add_parser("performance", help="Repeat a published fixed-rate performance measurement")
    _add_shared_arguments(performance)
    performance.add_argument("--rate", type=float, help="Requests per second; the reference's published level")
    performance.add_argument("--duration", type=float, help="Seconds of scheduled arrivals (default 60)")
    performance.add_argument("--warmup", type=float, help="Seconds of unrecorded warmup (default 30; 0 skips it)")
    latency = commands.add_parser("latency", help="Time one request at a time beside a per-request time on the website")
    latency.add_argument("--page", required=True, choices=sorted(latency_mode.PAGES), help="The website page")
    latency.add_argument("--endpoint", required=True, help="Your SIE server, for example http://localhost:8080")
    return parser


def plan_quality(args: argparse.Namespace) -> QualityPlan:
    if args.reference is not None:
        reference = json.loads(Path(args.reference).read_text(encoding="utf-8"))
        return _plan_from_reference(args, reference)
    return _plan_from_flags(args)


def _endpoint_key(url: str) -> tuple[str, str | None, int, str]:
    try:
        parts = urlsplit(url)
        port = parts.port
    except ValueError:
        raise ValueError("endpoint must be an http(s) URL") from None
    scheme = parts.scheme.lower()
    if port is None:
        port = 443 if scheme == "https" else 80
    elif port == 0:
        raise ValueError("endpoint must be an http(s) URL")
    return scheme, parts.hostname, port, parts.path.rstrip("/")


def normalize_endpoint(url: str) -> str:
    scheme, hostname, port, path = _endpoint_key(url)
    if scheme not in {"http", "https"} or hostname is None:
        raise ValueError("endpoint must be an http(s) URL")
    default_port = 443 if scheme == "https" else 80
    host = f"[{hostname}]" if ":" in hostname else hostname
    suffix = "" if port == default_port else f":{port}"
    return f"{scheme}://{host}{suffix}{path}"


def _key_scope() -> tuple[str, str]:
    key = os.environ.get("SIE_API_KEY", "").strip()
    scope = os.environ.get("SIE_BASE_URL", "").strip() or DEFAULT_ENDPOINT
    return key, scope


def api_key_for(endpoint: str) -> str:
    """Return SIE_API_KEY only for the origin it is scoped to, as the SDK does; the path plays no part."""
    key, scope = _key_scope()
    try:
        scope = normalize_endpoint(scope)
    except ValueError:
        return ""
    return key if key and _endpoint_key(scope)[:3] == _endpoint_key(endpoint)[:3] else ""


def _resolve_api_key(parser: argparse.ArgumentParser, endpoint: str) -> str:
    api_key = api_key_for(endpoint)
    if not api_key and endpoint == DEFAULT_ENDPOINT:
        parser.error(f"SIE_API_KEY must be set in the environment for {DEFAULT_ENDPOINT}")
    key, scope = _key_scope()
    if key and not api_key:
        print(
            f"note: SIE_API_KEY is scoped to {_scope_text(scope)} and was not sent to {endpoint}; "
            "set SIE_BASE_URL to the endpoint to send it",
            file=sys.stderr,
        )
    return api_key


def _without_credentials(url: str) -> str:
    parts = urlsplit(url)
    return urlunsplit((parts.scheme, parts.netloc.rpartition("@")[2], parts.path, "", ""))


def _scope_text(scope: str) -> str:
    try:
        normalize_endpoint(scope)
    except ValueError:
        return "an SIE_BASE_URL that is not an http(s) URL"
    return _without_credentials(scope)


def _endpoint_reasons(endpoint: str) -> list[str]:
    if _endpoint_key(endpoint) == _endpoint_key(DEFAULT_ENDPOINT):
        return []
    return [f"endpoint {_without_credentials(endpoint)} != {DEFAULT_ENDPOINT}"]


def _check_limit(params: Mapping[str, str]) -> None:
    if "limit" in params and (not params["limit"].isdigit() or int(params["limit"]) < 1):
        raise ValueError("limit must be a positive integer")


def _check_positive_integer(params: Mapping[str, str], name: str) -> None:
    if name in params and (not params[name].isdigit() or int(params[name]) < 1):
        raise ValueError(f"{name} must be a positive integer")


def _check_performance_params(task_name: str, params: Mapping[str, str], operation: str) -> None:
    unknown = sorted(set(params) - PERFORMANCE_PARAMETERS)
    if unknown:
        raise ValueError(
            f"a performance task key takes {', '.join(sorted(PERFORMANCE_PARAMETERS))}, got {', '.join(unknown)}"
        )
    _check_limit(params)
    _check_positive_integer(params, "candidates-k")
    if int(params.get("candidates-k", 1)) > SCORE_ITEMS_PER_REQUEST:
        raise ValueError(f"candidates-k must be at most {SCORE_ITEMS_PER_REQUEST}; a performance request is one call")
    split = eval_split(task_name)
    if params.get("split", split) != split:
        raise ValueError(f"only split={split} is available")
    if params.get("filter", "nonempty") != "nonempty":
        raise ValueError("filter must be nonempty")
    role = params.get("role", "queries")
    if role not in ROLES:
        raise ValueError("role must be queries or corpus")
    if operation == "score" and role != "queries":
        raise ValueError("the score task sends queries; role=corpus is not available")


def _check_timing(rate: float, duration_s: float, warmup_s: float) -> None:
    if not math.isfinite(rate) or rate <= 0:
        raise ValueError("--rate must be a positive finite number")
    if not math.isfinite(duration_s) or duration_s <= 0:
        raise ValueError("--duration must be a positive finite number")
    if not math.isfinite(warmup_s) or warmup_s < 0:
        raise ValueError("--warmup must be a finite number that is not negative")


def _refuse_fixed_flags(args: argparse.Namespace) -> None:
    for flag, value in (
        ("--model", args.model),
        ("--output", args.output),
        ("--items-per-request", args.items_per_request),
    ):
        if value is not None:
            raise ValueError(f"{flag} cannot be combined with --reference; the reference fixes it")


def _plain_reference_id(reference: Mapping[str, Any]) -> str:
    reference_id = reference["reference_id"]
    if not reference_id or Path(reference_id).name != reference_id or reference_id in {".", ".."}:
        raise ValueError(f"reference_id {reference_id!r} is not a plain file name")
    return reference_id


def _client_processes_reasons(reference: Mapping[str, Any]) -> list[str]:
    processes = reference["client"]["worker_count"] * reference["client"]["processes_per_worker"]
    return [] if processes == 1 else [f"client processes 1 != {processes}"]


def _check_output_flags(args: argparse.Namespace, operation: str) -> None:
    if operation == "encode" and (args.output is None or args.items_per_request is None):
        raise ValueError("--output and --items-per-request are required for encode tasks")
    if operation == "score" and (args.output is not None or args.items_per_request is not None):
        raise ValueError("--output and --items-per-request are not accepted for the score task")
    if args.items_per_request is not None and args.items_per_request < 1:
        raise ValueError("--items-per-request must be a positive integer")


def _supplied_baseline(args: argparse.Namespace) -> dict[str, Any] | None:
    if args.baseline is None and args.baseline_source is None:
        return None
    if args.baseline is None or args.baseline_source is None or not args.baseline_source.strip():
        raise ValueError("--baseline and --baseline-source go together")
    if not args.baseline_source.isprintable():
        raise ValueError("--baseline-source must be one line of printable text")
    metric, _, value = args.baseline.partition("=")
    if metric not in BASELINE_METRICS:
        raise ValueError(f"--baseline names {metric!r}; known metrics: {', '.join(sorted(BASELINE_METRICS))}")
    try:
        parsed = float(value)
    except ValueError:
        raise ValueError(f"--baseline must be <metric>=<value>, got {args.baseline!r}") from None
    return {
        "kind": "supplied",
        "metric": metric,
        "value": checked_baseline_value(parsed),
        "source": args.baseline_source,
    }


def _published_baseline(reference: Mapping[str, Any]) -> dict[str, Any]:
    for row in reference["metrics"]:
        if row["name"] == PRIMARY_METRIC:
            return {
                "kind": "published",
                "metric": PRIMARY_METRIC,
                "value": checked_baseline_value(float(row["value"])),
                "source": reference["reference_id"],
            }
    raise ValueError(f"the reference has no {PRIMARY_METRIC} metric")


def _checked_instruction(args: argparse.Namespace, operation: str) -> str | None:
    if args.instruction is None:
        return None
    if operation == "score":
        raise ValueError("--instruction is not accepted for the score task")
    if not args.instruction.strip():
        raise ValueError("--instruction must not be empty")
    if not args.instruction.isprintable():
        raise ValueError("--instruction must be one line of printable text")
    return args.instruction


def _checked_two_stage(task_name: str, params: Mapping[str, str]) -> tuple[str | None, int | None]:
    model, k = params.get("candidates_model"), params.get("candidates_k")
    if model is None and k is None:
        return None, None
    if model is None or k is None:
        raise ValueError("candidates_model and candidates_k go together")
    if task_name in SCORE_TASKS:
        raise ValueError(f"candidates_model and candidates_k are for the encode tasks; {task_name} is a score task")
    if not MODEL_ID.fullmatch(model):
        raise ValueError("candidates_model must be a plain <org>/<name> model id, with no profile")
    _check_positive_integer(params, "candidates_k")
    return model, int(k)


def _override(value: float | None, fixed: float, label: str, reasons: list[str]) -> float:
    if value is None or value == fixed:
        return fixed
    reasons.append(f"{label} {value:g} != {fixed:g}")
    return value


def _plan_from_reference(args: argparse.Namespace, reference: Mapping[str, Any]) -> QualityPlan:
    if reference.get("kind") != REFERENCE_KIND or reference.get("evaluation_mode") != "quality":
        raise ValueError("--reference must name a quality published-benchmark-reference file")
    _refuse_fixed_flags(args)
    for flag, value in (
        ("--baseline", args.baseline),
        ("--baseline-source", args.baseline_source),
        ("--instruction", args.instruction),
    ):
        if value is not None:
            raise ValueError(f"{flag} cannot be combined with --reference")
    reference_id = _plain_reference_id(reference)
    task_key = reference["task_key"]
    task_name, params = parse_task_key(task_key)
    reasons: list[str] = []
    if args.task is not None and args.task != task_key:
        smoke_name, smoke_params = parse_task_key(args.task)
        if smoke_name != task_name or set(smoke_params) != {"limit"}:
            raise ValueError(f"--task must equal the reference task_key {task_key!r} or be {task_name}?limit=N")
        task_key, params = args.task, smoke_params
        reasons.append(f"smoke key {task_key}")
    _check_limit(params)
    reasons.extend(_endpoint_reasons(args.endpoint))
    reasons.extend(_client_processes_reasons(reference))
    return QualityPlan(
        reference_id=reference_id,
        reference_metrics=list(reference["metrics"]),
        model=reference["model"],
        profile=reference["profile"],
        operation=reference["operation"],
        task_key=task_key,
        task_name=task_name,
        task_params=params,
        revision=reference["dataset"]["revision"],
        output_contract=reference["payload"]["output_contract"],
        items_per_request=reference["payload"]["items_per_request"],
        request_timeout_s=float(reference["timing"]["request_timeout_s"]),
        endpoint=normalize_endpoint(args.endpoint),
        model_version=reference["model_version"],
        comparable_reasons=reasons,
        baseline=_published_baseline(reference),
    )


def _plan_from_flags(args: argparse.Namespace) -> QualityPlan:
    if args.model is None or args.task is None:
        raise ValueError("--model and --task are required without --reference")
    task_name, params = parse_task_key(args.task)
    other_params = sorted(set(params) - QUALITY_PARAMETERS)
    if other_params:
        raise ValueError(
            f"a quality task key takes only limit, got {', '.join(other_params)}; "
            "a two-stage key adds candidates_model and candidates_k"
        )
    _check_limit(params)
    candidates_model, candidates_k = _checked_two_stage(task_name, params)
    model, _, profile = args.model.partition(":")
    operation = "score" if task_name in SCORE_TASKS or candidates_model is not None else "encode"
    _check_output_flags(args, operation)
    if args.output == "dense" and profile.casefold() == MUVERA_PROFILE:
        raise ValueError(
            "--output dense is refused for a muvera profile: the published method scores these vectors by the "
            "similarity the profile declares, usually the raw dot product, and this example scores dense only by cosine"
        )
    baseline = _supplied_baseline(args)
    instruction = _checked_instruction(args, operation)
    reasons = ["no reference"]
    if "limit" in params:
        reasons.append(f"smoke key {args.task}")
    reasons.extend(_endpoint_reasons(args.endpoint))
    return QualityPlan(
        reference_id=None,
        reference_metrics=[],
        model=model,
        profile=profile or "default",
        operation=operation,
        task_key=args.task,
        task_name=task_name,
        task_params=params,
        revision=HF_DATASETS[task_name][1],
        output_contract=args.output if operation == "encode" else "scores",
        items_per_request=args.items_per_request,
        request_timeout_s=DEFAULT_TIMEOUT_S,
        endpoint=normalize_endpoint(args.endpoint),
        model_version=None,
        comparable_reasons=reasons,
        baseline=baseline,
        instruction=instruction,
        candidates_model=candidates_model,
        candidates_k=candidates_k,
    )


def plan_performance(args: argparse.Namespace) -> PerformancePlan:
    if args.reference is not None:
        reference = json.loads(Path(args.reference).read_text(encoding="utf-8"))
        return _performance_plan_from_reference(args, reference)
    return _performance_plan_from_flags(args)


def _performance_plan_from_reference(args: argparse.Namespace, reference: Mapping[str, Any]) -> PerformancePlan:
    if reference.get("kind") != REFERENCE_KIND or reference.get("evaluation_mode") != PERFORMANCE_MODE:
        raise ValueError("--reference must name a performance published-benchmark-reference file")
    _refuse_fixed_flags(args)
    reference_id = _plain_reference_id(reference)
    task_key = reference["task_key"]
    if args.task is not None and args.task != task_key:
        raise ValueError(f"--task must equal the reference task_key {task_key!r}")
    task_name, params = parse_task_key(task_key)
    operation = reference["operation"]
    _check_performance_params(task_name, params, operation)
    level, timing = reference["schedule"]["levels"][0], reference["timing"]
    reasons: list[str] = []
    rate = _override(args.rate, float(level["rate"]), "rate", reasons)
    duration_s = _override(args.duration, float(level["duration_s"]), "duration", reasons)
    warmup_s = _override(args.warmup, float(timing["warmup_s"]), "warmup", reasons)
    _check_timing(rate, duration_s, warmup_s)
    reasons.extend(_endpoint_reasons(args.endpoint))
    reasons.extend(_client_processes_reasons(reference))
    readiness_timeout_s = timing["target_readiness_timeout_s"]
    return PerformancePlan(
        reference_id=reference_id,
        reference_metrics=list(reference["metrics"]),
        model=reference["model"],
        profile=reference["profile"],
        operation=operation,
        task_key=task_key,
        task_name=task_name,
        task_params=params,
        revision=reference["dataset"]["revision"],
        output_contract=reference["payload"]["output_contract"],
        items_per_request=reference["payload"]["items_per_request"],
        rate=rate,
        duration_s=duration_s,
        warmup_s=warmup_s,
        settle_s=float(timing["settle_s"]),
        drain_s=float(timing["drain_s"]),
        request_timeout_s=float(timing["request_timeout_s"]),
        target_readiness_timeout_s=(
            DEFAULT_READINESS_TIMEOUT_S if readiness_timeout_s is None else float(readiness_timeout_s)
        ),
        max_in_flight=int(timing["max_in_flight_per_process"]),
        lag_tolerance_ms=float(timing["scheduling_lag_tolerance_ms"]),
        endpoint=normalize_endpoint(args.endpoint),
        model_version=reference["model_version"],
        comparable_reasons=reasons,
    )


def _performance_plan_from_flags(args: argparse.Namespace) -> PerformancePlan:
    if args.model is None or args.task is None:
        raise ValueError("--model and --task are required without --reference")
    if args.rate is None:
        raise ValueError("--rate is required without --reference")
    task_name, params = parse_task_key(args.task)
    model, _, profile = args.model.partition(":")
    operation = "score" if task_name in SCORE_TASKS else "encode"
    _check_performance_params(task_name, params, operation)
    _check_output_flags(args, operation)
    duration_s = DEFAULT_DURATION_S if args.duration is None else args.duration
    warmup_s = DEFAULT_WARMUP_S if args.warmup is None else args.warmup
    _check_timing(args.rate, duration_s, warmup_s)
    request_timeout_s, max_in_flight = PERFORMANCE_LIMITS[operation]
    return PerformancePlan(
        reference_id=None,
        reference_metrics=[],
        model=model,
        profile=profile or "default",
        operation=operation,
        task_key=args.task,
        task_name=task_name,
        task_params=params,
        revision=HF_DATASETS[task_name][1],
        output_contract=args.output if operation == "encode" else "scores",
        items_per_request=args.items_per_request,
        rate=args.rate,
        duration_s=duration_s,
        warmup_s=warmup_s,
        settle_s=DEFAULT_SETTLE_S,
        drain_s=DEFAULT_DRAIN_S,
        request_timeout_s=request_timeout_s,
        target_readiness_timeout_s=DEFAULT_READINESS_TIMEOUT_S,
        max_in_flight=max_in_flight,
        lag_tolerance_ms=DEFAULT_LAG_TOLERANCE_MS,
        endpoint=normalize_endpoint(args.endpoint),
        model_version=None,
        comparable_reasons=["no reference", *_endpoint_reasons(args.endpoint)],
    )


def select_queries(queries: Queries, qrels: Qrels, params: Mapping[str, str]) -> Queries:
    selected = {query_id: text for query_id, text in queries.items() if qrels.get(query_id)}
    if "limit" not in params:
        return selected
    limit = int(params["limit"])
    return dict(list(selected.items())[:limit])


def load_quality_data(plan: QualityPlan, loaders: Loaders | None = None) -> QualityData:
    loaders = loaders or Loaders()
    qrels = loaders.qrels(plan.task_name, plan.revision)
    queries = select_queries(loaders.queries(plan.task_name, plan.revision), qrels, plan.task_params)
    top_ranked = None
    if plan.operation == "score" and plan.candidates_model is None:
        candidates = loaders.top_ranked(plan.task_name, plan.revision)
        top_ranked = {query_id: list(candidates[query_id]) for query_id in queries}
    return QualityData(
        corpus=loaders.corpus(plan.task_name, plan.revision),
        queries=queries,
        qrels={query_id: qrels[query_id] for query_id in queries},
        top_ranked=top_ranked,
    )


def document_text(document: Mapping[str, str]) -> str:
    title = document.get("title") or ""
    return f"{title} {document['text']}".strip() if title else document["text"].strip()


def performance_text(document: Mapping[str, str]) -> str:
    title = document.get("title") or ""
    return f"{title}\n{document['text']}" if title else document["text"]


def select_ids(texts: Mapping[str, str], params: Mapping[str, str]) -> list[str]:
    nonempty = params.get("filter") == "nonempty"
    selected = sorted(item_id for item_id, text in texts.items() if not nonempty or text.strip())
    return selected[: int(params["limit"])] if "limit" in params else selected


def load_performance_data(plan: PerformancePlan, loaders: Loaders | None = None) -> PerformanceData:
    loaders = loaders or Loaders()
    params = plan.task_params
    if plan.operation == "score":
        queries = loaders.queries(plan.task_name, plan.revision)
        corpus = loaders.corpus(plan.task_name, plan.revision)
        top_ranked = loaders.top_ranked(plan.task_name, plan.revision)
        selected = select_ids(queries, params)
        k = int(params.get("candidates-k", SCORE_ITEMS_PER_REQUEST))
        requests: list[ScoreRequest] = [
            (
                {"text": queries[query_id]},
                [{"id": doc_id, "text": performance_text(corpus[doc_id])} for doc_id in top_ranked[query_id][:k]],
            )
            for query_id in selected
        ]
        return PerformanceData(selected, requests)
    if params.get("role", "queries") == "corpus":
        corpus = loaders.corpus(plan.task_name, plan.revision)
        selected = select_ids({doc_id: document["text"] for doc_id, document in corpus.items()}, params)
        return PerformanceData(
            selected, [{"id": doc_id, "text": performance_text(corpus[doc_id])} for doc_id in selected]
        )
    queries = loaders.queries(plan.task_name, plan.revision)
    judged = loaders.qrels(plan.task_name, plan.revision)
    selected = select_ids({query_id: text for query_id, text in queries.items() if query_id in judged}, params)
    return PerformanceData(selected, [{"id": query_id, "text": queries[query_id]} for query_id in selected])


def load_level_name(rate: float) -> str:
    return f"fixed-{rate:g}-rps".replace(".", "-")


def schedule_offsets(rate: float, duration_s: float) -> list[Fraction]:
    exact_rate, exact_duration = Fraction(str(rate)), Fraction(str(duration_s))
    return [Fraction(index) / exact_rate for index in range(math.ceil(exact_rate * exact_duration))]


def request_rows(sequence: int, items_per_request: int, length: int) -> list[int]:
    return [(sequence * items_per_request + position) % length for position in range(items_per_request)]


def request_payload(plan: PerformancePlan, data: PerformanceData, sequence: int) -> Any:
    if not data.payloads:
        raise ValueError(f"the task key {plan.task_key!r} selects no rows")
    if plan.operation == "score":
        return data.payloads[sequence % len(data.payloads)]
    return [data.payloads[row] for row in request_rows(sequence, plan.items_per_request or 1, len(data.payloads))]


def performance_call(
    client: Any, plan: PerformancePlan, payload: Any, *, wait_for_capacity: bool, provision_timeout_s: float
) -> Any:
    if plan.operation == "score":
        query, items = payload
        return client.score(
            plan.route,
            query,
            items,
            wait_for_capacity=wait_for_capacity,
            provision_timeout_s=provision_timeout_s,
            max_oom_retries=0,
        )
    return client.encode(
        plan.route,
        payload,
        output_types=[plan.output_contract],
        is_query=False,
        wait_for_capacity=wait_for_capacity,
        provision_timeout_s=provision_timeout_s,
        max_oom_retries=0,
    )


def request_status(exc: BaseException) -> str:
    cause: BaseException | None = exc
    while cause is not None:
        if isinstance(cause, httpx.TimeoutException):
            return "timeout"
        cause = cause.__cause__
    return f"error:{type(exc).__name__}"


@dataclass
class Dispatcher:
    plan: PerformancePlan
    data: PerformanceData
    client: Any
    log: CallLog = field(default_factory=CallLog)
    rows: list[dict[str, Any]] = field(default_factory=list)
    readiness_failure: str | None = None
    in_flight_peak: int = 0
    _in_flight: int = 0
    _origin_ns: int = 0
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def send(self, sequence: int, offset: Fraction) -> dict[str, Any]:
        payload = request_payload(self.plan, self.data, sequence)
        with self._lock:
            in_flight_at_send = self._in_flight
            self._in_flight += 1
            self.in_flight_peak = max(self.in_flight_peak, self._in_flight)
        failure: Exception | None = None
        sent_ns = time.perf_counter_ns() - self._origin_ns
        try:
            performance_call(
                self.client,
                self.plan,
                payload,
                wait_for_capacity=False,
                provision_timeout_s=self.plan.request_timeout_s,
            )
        except Exception as exc:  # noqa: BLE001
            failure = exc
        completed_ns = time.perf_counter_ns() - self._origin_ns
        retry_count = int(self.client.last_retry_count)
        with self._lock:
            self._in_flight -= 1
            if failure is None:
                self.log.record(self.client)
            else:
                self.log.fail(self.client, failure)
        return {
            "sequence": sequence,
            "level_name": self.plan.level_name,
            "offset_s": float(offset),
            "actual_send_offset_ns": sent_ns,
            "completed_offset_ns": completed_ns,
            "scheduling_lag_ms": max(0.0, (sent_ns - nanoseconds(offset)) / NS_PER_MS),
            "status": "success" if failure is None else request_status(failure),
            "attempt": 1,
            "client_retry_count": retry_count,
            "in_flight_at_send": in_flight_at_send,
        }

    def run(self) -> None:
        offsets = schedule_offsets(self.plan.rate, self.plan.duration_s)
        with ThreadPoolExecutor(max_workers=self.plan.max_in_flight) as pool:
            self._origin_ns = time.perf_counter_ns()
            futures = []
            for sequence, offset in enumerate(offsets):
                remaining_ns = self._origin_ns + nanoseconds(offset) - time.perf_counter_ns()
                if remaining_ns > 0:
                    time.sleep(remaining_ns / NS_PER_S)
                futures.append(pool.submit(self.send, sequence, offset))
            _, pending = wait(futures, timeout=self.plan.drain_s)
            wait(pending)
        self.rows = [future.result() for future in futures]


def readiness_request(client: Any, plan: PerformancePlan, data: PerformanceData) -> None:
    payload = request_payload(plan, data, 0)
    performance_call(client, plan, payload, wait_for_capacity=True, provision_timeout_s=plan.target_readiness_timeout_s)


def warmup(client: Any, plan: PerformancePlan, data: PerformanceData) -> None:
    payload = request_payload(plan, data, 0)
    deadline = time.monotonic() + plan.warmup_s
    while time.monotonic() < deadline:
        with contextlib.suppress(Exception):
            performance_call(client, plan, payload, wait_for_capacity=False, provision_timeout_s=plan.request_timeout_s)


def encoder_plan(plan: QualityPlan) -> QualityPlan:
    return replace(
        plan,
        model=plan.candidates_model,
        profile="default",
        operation="encode",
        output_contract="dense",
        items_per_request=CANDIDATE_ENCODE_ITEMS_PER_REQUEST,
    )


def quality_readiness_request(client: Any, plan: QualityPlan, data: QualityData) -> None:
    if plan.candidates_model is not None:
        _encode_readiness_request(client, encoder_plan(plan), data)
    if plan.operation == "score":
        query_id, text = next(iter(data.queries.items()))
        if plan.candidates_model is not None:
            candidates = list(data.corpus)[: plan.candidates_k]
        else:
            candidates = (data.top_ranked or {})[query_id]
        items = [
            {"id": doc_id, "text": document_text(data.corpus[doc_id])}
            for doc_id in candidates[:SCORE_ITEMS_PER_REQUEST]
        ]
        client.score(
            plan.route,
            {"text": text},
            items,
            wait_for_capacity=True,
            provision_timeout_s=DEFAULT_READINESS_TIMEOUT_S,
            max_oom_retries=0,
        )
        return
    _encode_readiness_request(client, plan, data)


def _encode_readiness_request(client: Any, plan: QualityPlan, data: QualityData) -> None:
    document_ids = list(data.corpus)[: plan.items_per_request]
    client.encode(
        plan.route,
        [{"id": doc_id, "text": document_text(data.corpus[doc_id])} for doc_id in document_ids],
        output_types=[plan.output_contract],
        is_query=False,
        wait_for_capacity=True,
        provision_timeout_s=DEFAULT_READINESS_TIMEOUT_S,
        max_oom_retries=0,
    )


def _call(log: CallLog, client: Any, method: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
    try:
        result = method(*args, **kwargs)
    except Exception as exc:
        log.fail(client, exc)
        raise RequestFailed(str(exc)) from exc
    log.record(client)
    return result


def encode_texts(
    client: Any, plan: QualityPlan, ids: Sequence[str], texts: Sequence[str], log: CallLog, *, is_query: bool
) -> list[Any]:
    size = plan.items_per_request or 0
    if size < 1:
        raise ValueError("encode tasks need a positive items_per_request")
    instruction = {"instruction": plan.instruction} if is_query and plan.instruction is not None else {}
    results: list[Any] = []
    for start in range(0, len(texts), size):
        batch_ids, batch_texts = ids[start : start + size], texts[start : start + size]
        items = [{"id": item_id, "text": text} for item_id, text in zip(batch_ids, batch_texts, strict=True)]
        batch = _call(
            log,
            client,
            client.encode,
            plan.route,
            items,
            output_types=[plan.output_contract],
            is_query=is_query,
            wait_for_capacity=False,
            provision_timeout_s=plan.request_timeout_s,
            max_oom_retries=0,
            **instruction,
        )
        if len(batch) != len(items):
            raise ValueError("encode returned a different number of results than items sent")
        results.extend(batch)
    return results


def sparse_columns(*groups: Sequence[Mapping[str, Any]]) -> int:
    highest = -1
    for group in groups:
        for result in group:
            indices = np.asarray(result["sparse"]["indices"])
            if indices.size:
                highest = max(highest, int(indices.max()))
    return highest + 1


def sparse_matrix(results: Sequence[Mapping[str, Any]], columns: int) -> csr_matrix:
    indices = [np.asarray(result["sparse"]["indices"], dtype=np.int32) for result in results]
    values = [np.asarray(result["sparse"]["values"], dtype=np.float32) for result in results]
    indptr = np.zeros(len(results) + 1, dtype=np.int64)
    indptr[1:] = np.cumsum([len(row) for row in indices])
    data = np.concatenate(values) if values else np.zeros(0, dtype=np.float32)
    cols = np.concatenate(indices) if indices else np.zeros(0, dtype=np.int32)
    return csr_matrix((data, cols, indptr), shape=(len(results), columns), dtype=np.float32)


def sparse_scores(queries: Sequence[Mapping[str, Any]], documents: Sequence[Mapping[str, Any]]) -> np.ndarray:
    columns = sparse_columns(queries, documents)
    return (sparse_matrix(queries, columns) @ sparse_matrix(documents, columns).T).toarray().astype(np.float32)


def _rows_and_norms(vectors: Any) -> tuple[np.ndarray, np.ndarray]:
    rows = np.asarray(vectors, dtype=np.float32)
    if rows.ndim != 2 or rows.shape[0] == 0:
        raise ValueError("expected a non-empty two-dimensional array of vectors")
    return rows, np.linalg.norm(rows, axis=1, keepdims=True)


def normalized_rows(vectors: Any) -> np.ndarray:
    rows, norms = _rows_and_norms(vectors)
    return rows / np.where(norms > 0, norms, 1.0)


def _cosine_rows(vectors: Any) -> np.ndarray:
    rows, norms = _rows_and_norms(vectors)
    return rows / np.maximum(norms, COSINE_NORM_FLOOR)


def dense_scores(queries: Sequence[Any], documents: Sequence[Any]) -> np.ndarray:
    return _cosine_rows(queries) @ _cosine_rows(documents).T


def _query_groups(query_rows: Sequence[np.ndarray]) -> list[tuple[int, int]]:
    groups: list[tuple[int, int]] = []
    start, rows = 0, 0
    for end, query in enumerate(query_rows, start=1):
        rows += len(query)
        if rows >= QUERY_TOKEN_ROWS or end == len(query_rows):
            groups.append((start, end))
            start, rows = end, 0
    return groups


def _starts(parts: Sequence[np.ndarray]) -> np.ndarray:
    return np.cumsum([0] + [len(part) for part in parts[:-1]], dtype=np.int64)


def maxsim_scores(queries: Sequence[Any], documents: Sequence[Any]) -> np.ndarray:
    query_rows = [normalized_rows(query) for query in queries]
    groups = [
        (start, end, np.concatenate(query_rows[start:end]), _starts(query_rows[start:end]))
        for start, end in _query_groups(query_rows)
    ]
    scores = np.zeros((len(queries), len(documents)), dtype=np.float32)
    for chunk_start in range(0, len(documents), DOCUMENT_CHUNK):
        chunk = [normalized_rows(document) for document in documents[chunk_start : chunk_start + DOCUMENT_CHUNK]]
        tokens = np.concatenate(chunk)
        document_starts = _starts(chunk)
        for start, end, group_tokens, query_starts in groups:
            per_document = np.maximum.reduceat(group_tokens @ tokens.T, document_starts, axis=1)
            scores[start:end, chunk_start : chunk_start + len(chunk)] = np.add.reduceat(
                per_document, query_starts, axis=0
            )
    return scores


def top_k_predictions(
    scores: np.ndarray, query_ids: Sequence[str], document_ids: Sequence[str], k: int = TOP_K
) -> PredictionMap:
    predictions: PredictionMap = {}
    for row, query_id in zip(scores, query_ids, strict=True):
        order = np.argsort(-row, kind="stable")[:k]
        predictions[query_id] = {document_ids[index]: float(row[index]) for index in order}
    return predictions


def encode_predictions(
    client: Any, plan: QualityPlan, data: QualityData, log: CallLog, *, k: int = TOP_K
) -> PredictionMap:
    document_ids = list(data.corpus)
    document_texts = [document_text(data.corpus[doc_id]) for doc_id in document_ids]
    documents = encode_texts(client, plan, document_ids, document_texts, log, is_query=False)
    query_ids = list(data.queries)
    queries = encode_texts(client, plan, query_ids, [data.queries[q] for q in query_ids], log, is_query=True)
    if plan.output_contract == "sparse":
        scores = sparse_scores(queries, documents)
    elif plan.output_contract == "dense":
        query_vectors, document_vectors = [q["dense"] for q in queries], [d["dense"] for d in documents]
        for role, vectors in (("document", document_vectors), ("query", query_vectors)):
            if not any(np.any(vector) for vector in vectors):
                raise ValueError(f"every returned dense {role} vector is all zeros")
        scores = dense_scores(query_vectors, document_vectors)
    else:
        scores = maxsim_scores([q["multivector"] for q in queries], [d["multivector"] for d in documents])
    return top_k_predictions(scores, query_ids, document_ids, k)


def candidate_batches(candidates: Sequence[str]) -> list[Sequence[str]]:
    return [
        candidates[start : start + SCORE_ITEMS_PER_REQUEST]
        for start in range(0, len(candidates), SCORE_ITEMS_PER_REQUEST)
    ]


def rerank_predictions(client: Any, plan: QualityPlan, data: QualityData, log: CallLog) -> PredictionMap:
    top_ranked = data.top_ranked or {}
    predictions: PredictionMap = {}
    for query_id, text in data.queries.items():
        scored: dict[str, float] = {}
        for candidates in candidate_batches(top_ranked[query_id]):
            items = [{"id": doc_id, "text": document_text(data.corpus[doc_id])} for doc_id in candidates]
            result = _call(
                log,
                client,
                client.score,
                plan.route,
                {"text": text},
                items,
                wait_for_capacity=False,
                provision_timeout_s=plan.request_timeout_s,
                max_oom_retries=0,
            )
            batch = {entry["item_id"]: float(entry["score"]) for entry in result["scores"]}
            if set(batch) != set(candidates):
                raise ValueError(f"score returned item ids that differ from the candidates of query {query_id}")
            scored.update(batch)
        predictions[query_id] = scored
    return predictions


def two_stage_predictions(client: Any, plan: QualityPlan, data: QualityData, log: CallLog) -> PredictionMap:
    retrieved = encode_predictions(client, encoder_plan(plan), data, log, k=plan.candidates_k or TOP_K)
    log.encoder_revisions, log.execution_revisions = log.execution_revisions, []
    top_ranked = {query_id: list(documents) for query_id, documents in retrieved.items()}
    return rerank_predictions(client, plan, replace(data, top_ranked=top_ranked), log)


def client_description() -> str:
    return f"single process, {platform.platform()}, python {platform.python_version()}, sie-sdk {sie_sdk.__version__}"


def _timestamp(moment: datetime) -> str:
    return moment.strftime("%Y-%m-%dT%H:%M:%SZ")


def _slug(route: str, task_key: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "-", f"{route}-{task_key}").strip("-")


def _common_flags(plan: QualityPlan | PerformancePlan) -> list[str]:
    flags = ["--model", plan.route, "--task", plan.task_key]
    if plan.operation == "encode":
        flags += ["--output", plan.output_contract, "--items-per-request", str(plan.items_per_request)]
    return flags


def _endpoint_flags(endpoint: str) -> list[str]:
    return [] if endpoint == DEFAULT_ENDPOINT else ["--endpoint", endpoint]


def quality_command_line(plan: QualityPlan) -> str:
    if plan.reference_id is not None:
        flags = ["--reference", f"references/{plan.reference_id}.json"]
        if "limit" in plan.task_params:
            flags += ["--task", plan.task_key]
    else:
        flags = _common_flags(plan)
        if plan.baseline is not None and plan.baseline["kind"] == "supplied":
            metric, value, source = plan.baseline["metric"], plan.baseline["value"], plan.baseline["source"]
            flags += ["--baseline", f"{metric}={value!r}", f"--baseline-source={source}"]
        if plan.instruction is not None:
            flags.append(f"--instruction={plan.instruction}")
    return shlex.join([*COMMAND_PREFIX, "quality", *flags, *_endpoint_flags(plan.endpoint)])


def performance_command_line(plan: PerformancePlan) -> str:
    if plan.reference_id is not None:
        flags = ["--reference", f"references/{plan.reference_id}.json"]
    else:
        flags = _common_flags(plan)
    flags += ["--rate", str(plan.rate), "--duration", str(plan.duration_s), "--warmup", str(plan.warmup_s)]
    return shlex.join([*COMMAND_PREFIX, "performance", *flags, *_endpoint_flags(plan.endpoint)])


def comparison_basis(reference_id: str | None, comparable: bool, reasons: Sequence[str]) -> str | None:
    if reference_id is None:
        return None
    if comparable:
        return "same parameters"
    return "same method, different parameters: " + "; ".join(reasons)


def _largest_score_request(plan: QualityPlan, data: QualityData) -> int:
    if plan.candidates_k is not None:
        largest = min(plan.candidates_k, len(data.corpus))
    else:
        largest = max((len(candidates) for candidates in (data.top_ranked or {}).values()), default=0)
    return min(largest, SCORE_ITEMS_PER_REQUEST)


def _payload_parameters(plan: QualityPlan) -> list[dict[str, Any]]:
    parameters: list[dict[str, Any]] = []
    if plan.instruction is not None:
        parameters.append({"name": "instruction", "value": plan.instruction})
    if plan.candidates_model is not None:
        parameters.append({"name": "candidates_model", "value": plan.candidates_model})
        parameters.append({"name": "candidates_k", "value": plan.candidates_k})
    return parameters


def build_record(
    plan: QualityPlan,
    data: QualityData,
    log: CallLog,
    started: datetime,
    finished: datetime,
    predictions_path: Path | None,
    rows: list[dict[str, Any]],
    official_lookup: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    reasons = list(plan.comparable_reasons)
    if log.revision_changed:
        reasons.append("execution revision changed")
    comparable = not reasons
    validity_reasons: list[str] = []
    if log.readiness_failure is not None:
        validity_reasons.append(f"readiness request failed: {log.readiness_failure}")
    if log.sdk_retries:
        validity_reasons.append(f"sdk retries {log.sdk_retries}")
    if log.failure is not None:
        validity_reasons.append(f"request failed: {log.failure}")
    record = {
        "kind": RESULT_KIND,
        "schema_version": 1,
        "reference_id": plan.reference_id,
        "comparable": comparable,
        "comparable_reasons": reasons,
        "evaluation_mode": "quality",
        "model": plan.model,
        "profile": plan.profile,
        "operation": plan.operation,
        "model_version": plan.model_version,
        "execution_revisions": list(dict.fromkeys([*log.encoder_revisions, *log.execution_revisions])),
        "ingress": {"url": plan.endpoint},
        "task_key": plan.task_key,
        "dataset": {
            "name": plan.task_name,
            "revision": plan.revision,
            "subset": eval_split(plan.task_name),
            "parameters": [{"name": name, "value": value} for name, value in plan.task_params.items()],
        },
        "payload": {
            "items_per_request": plan.items_per_request or _largest_score_request(plan, data),
            "output_contract": plan.output_contract,
            "parameters": _payload_parameters(plan),
        },
        "client": {"description": client_description(), "worker_count": 1, "processes_per_worker": 1},
        "scorer": {"id": SCORER_ID, "version": scorer_version(), "reference_scorer_matched": False},
        "run": {
            "started_at": _timestamp(started),
            "finished_at": _timestamp(finished),
            "endpoint": plan.endpoint,
            "request_count": log.request_count,
            "failure_count": log.failure_count,
            "predictions_path": str(predictions_path) if predictions_path is not None else None,
            "command": quality_command_line(plan),
        },
        "validity": {"valid": not validity_reasons, "reasons": validity_reasons, "sdk_retries": log.sdk_retries},
        "metrics": rows,
        "comparison": compare(rows, plan.reference_metrics) if plan.reference_id is not None else [],
        "comparison_basis": comparison_basis(plan.reference_id, comparable, reasons),
        "baseline": plan.baseline,
        "verdict": compute_verdict(
            {row["name"]: row["value"] for row in rows}, plan.baseline, limited="limit" in plan.task_params
        ),
    }
    if official_lookup is not None:
        record["official_lookup"] = dict(official_lookup)
    return record


def performance_validity(plan: PerformancePlan, dispatcher: Dispatcher, metrics: Mapping[str, float]) -> dict[str, Any]:
    log = dispatcher.log
    reasons: list[str] = []
    if dispatcher.readiness_failure is not None:
        reasons.append(f"readiness request failed: {dispatcher.readiness_failure}")
    if log.sdk_retries:
        reasons.append(f"sdk retries {log.sdk_retries}")
    lag_p99_ms = metrics.get("scheduling_lag_p99_ms")
    if lag_p99_ms is not None and lag_p99_ms > plan.lag_tolerance_ms:
        reasons.append(f"scheduling lag p99 {lag_p99_ms:.3f} ms > {plan.lag_tolerance_ms:g} ms")
    if dispatcher.in_flight_peak > plan.max_in_flight:
        reasons.append(f"in-flight peak {dispatcher.in_flight_peak} > {plan.max_in_flight}")
    if len(log.execution_revisions) > 1:
        reasons.append("execution revision changed")
    return {
        "valid": not reasons,
        "reasons": reasons,
        "lag_max_ms": max((row["scheduling_lag_ms"] for row in dispatcher.rows), default=0.0),
        "in_flight_peak": dispatcher.in_flight_peak,
        "timeouts": sum(row["status"] == "timeout" for row in dispatcher.rows),
        "sdk_retries": log.sdk_retries,
    }


def build_performance_record(
    plan: PerformancePlan,
    data: PerformanceData,
    dispatcher: Dispatcher,
    started: datetime,
    finished: datetime,
    requests_path: Path | None,
    rows: Rows,
) -> dict[str, Any]:
    log = dispatcher.log
    reasons = list(plan.comparable_reasons)
    if len(log.execution_revisions) > 1:
        reasons.append("execution revision changed")
    comparable = not reasons
    candidate_counts = [len(items) for _, items in data.payloads] if plan.operation == "score" else []
    return {
        "kind": RESULT_KIND,
        "schema_version": 1,
        "reference_id": plan.reference_id,
        "comparable": comparable,
        "comparable_reasons": reasons,
        "evaluation_mode": PERFORMANCE_MODE,
        "model": plan.model,
        "profile": plan.profile,
        "operation": plan.operation,
        "model_version": plan.model_version,
        "execution_revisions": list(log.execution_revisions),
        "ingress": {"url": plan.endpoint},
        "task_key": plan.task_key,
        "dataset": {
            "name": plan.task_name,
            "revision": plan.revision,
            "subset": eval_split(plan.task_name),
            "parameters": [{"name": name, "value": value} for name, value in plan.task_params.items()],
            "selected_ids": list(data.selected_ids),
        },
        "payload": {
            "items_per_request": plan.items_per_request or max(candidate_counts, default=0),
            "output_contract": plan.output_contract,
            "parameters": [{"name": "is_query", "value": False}] if plan.operation == "encode" else [],
        },
        "client": {"description": client_description(), "worker_count": 1, "processes_per_worker": 1},
        "load_level_name": plan.level_name,
        "schedule": {
            "arrival": "deterministic",
            "levels": [{"name": plan.level_name, "rate": plan.rate, "duration_s": plan.duration_s}],
        },
        "timing": {
            "warmup_s": plan.warmup_s,
            "settle_s": plan.settle_s,
            "drain_s": plan.drain_s,
            "request_timeout_s": plan.request_timeout_s,
            "target_readiness_timeout_s": plan.target_readiness_timeout_s,
            "max_in_flight_per_process": plan.max_in_flight,
            "scheduling_lag_tolerance_ms": plan.lag_tolerance_ms,
            "retry": dict(RETRY_POLICY),
        },
        "run": {
            "started_at": _timestamp(started),
            "finished_at": _timestamp(finished),
            "endpoint": plan.endpoint,
            "request_count": log.request_count,
            "failure_count": log.failure_count,
            "requests_path": str(requests_path) if requests_path is not None else None,
            "command": performance_command_line(plan),
        },
        "validity": performance_validity(plan, dispatcher, {row["name"]: row["value"] for row in rows}),
        "metrics": rows,
        "comparison": compare(rows, plan.reference_metrics) if plan.reference_id is not None else [],
        "comparison_basis": comparison_basis(plan.reference_id, comparable, reasons),
    }


def run_quality(
    plan: QualityPlan,
    data: QualityData,
    client: Any,
    *,
    results_dir: Path = RESULTS_DIR,
    now: Callable[[], datetime] | None = None,
    official_lookup: Mapping[str, str] | None = None,
) -> tuple[dict[str, Any], Path]:
    clock = now or (lambda: datetime.now(UTC))
    started = clock()
    log = CallLog()
    predictions: PredictionMap | None = None
    try:
        try:
            quality_readiness_request(client, plan, data)
        except Exception as exc:  # noqa: BLE001
            log.readiness_failure = f"{type(exc).__name__}: {exc}"
        if log.readiness_failure is None:
            if plan.candidates_model is not None:
                predictions = two_stage_predictions(client, plan, data, log)
            elif plan.operation == "score":
                predictions = rerank_predictions(client, plan, data, log)
            else:
                predictions = encode_predictions(client, plan, data, log)
    except (RequestFailed, ValueError, KeyError, TypeError) as exc:
        if not isinstance(exc, RequestFailed):
            log.failure = f"{type(exc).__name__}: {exc}"
        predictions = None
    finished = clock()
    stem = f"{plan.reference_id or _slug(plan.route, plan.task_key)}-{started.strftime('%Y%m%dT%H%M%SZ')}"
    results_dir.mkdir(parents=True, exist_ok=True)
    result_path = results_dir / f"{stem}.json"
    predictions_path = None
    rows: list[dict[str, Any]] = []
    if predictions is not None:
        predictions_path = results_dir / f"{stem}.predictions.json"
        write_json(predictions_path, predictions, compact=True)
        rows = metric_rows(task_metrics(predictions, data.qrels, plan.task_name))
    record = build_record(plan, data, log, started, finished, predictions_path, rows, official_lookup)
    write_json(result_path, record)
    return record, result_path


def run_performance(
    plan: PerformancePlan,
    data: PerformanceData,
    client: Any,
    *,
    results_dir: Path = RESULTS_DIR,
    now: Callable[[], datetime] | None = None,
) -> tuple[dict[str, Any], Path]:
    clock = now or (lambda: datetime.now(UTC))
    started = clock()
    dispatcher = Dispatcher(plan, data, client)
    try:
        readiness_request(client, plan, data)
    except Exception as exc:  # noqa: BLE001
        dispatcher.readiness_failure = f"{type(exc).__name__}: {exc}"
    if dispatcher.readiness_failure is None:
        warmup(client, plan, data)
        time.sleep(plan.settle_s)
        dispatcher.run()
    finished = clock()
    stem = f"{plan.reference_id or _slug(plan.route, plan.task_key)}-{started.strftime('%Y%m%dT%H%M%SZ')}"
    results_dir.mkdir(parents=True, exist_ok=True)
    result_path = results_dir / f"{stem}.json"
    requests_path = None
    rows: Rows = []
    if dispatcher.rows:
        requests_path = results_dir / f"{stem}.requests.jsonl"
        write_jsonl(requests_path, dispatcher.rows)
        rows = performance_metric_rows(performance_metrics(dispatcher.rows, plan.rate, plan.duration_s))
    record = build_performance_record(plan, data, dispatcher, started, finished, requests_path, rows)
    write_json(result_path, record)
    return record, result_path


def _plan_skip_reason(plan: QualityPlan) -> str | None:
    if "limit" in plan.task_params:
        return "a task key with limit is not compared with a full-dataset number"
    if plan.candidates_model is not None:
        return "the results repository has no number for a retriever and reranker pair"
    if plan.profile != "default":
        return "the results repository has no number for a profile"
    return None


def _served_model(client: Any, model: str) -> tuple[set[str] | None, str | None]:
    try:
        info = client.get_model(model)
        outputs, revision = info.get("outputs"), info.get("revision")
        known = set(ENCODE_OUTPUTS).intersection(outputs) if isinstance(outputs, list) and outputs else None
    except Exception:  # noqa: BLE001
        return None, None
    return known, plain_text(revision)


def _output_skip_reason(plan: QualityPlan, outputs: set[str] | None) -> str | None:
    if plan.operation != "encode" or plan.output_contract == "dense":
        return None
    if outputs is None:
        return f"the endpoint did not say what the model outputs, and an official number is for dense, not {plan.output_contract}"
    if len(outputs) > 1:
        return f"the model has several outputs and the official number is for dense, not {plan.output_contract}"
    return None


def _official_lookup(plan: QualityPlan, client: Any, fetch: Fetch) -> tuple[Lookup, str | None]:
    reason = _plan_skip_reason(plan)
    if reason is not None:
        return Lookup(SKIPPED, reason), None
    outputs, served_revision = _served_model(client, plan.model)
    reason = _output_skip_reason(plan, outputs)
    if reason is not None:
        return Lookup(SKIPPED, reason), served_revision
    lookup = lookup_official(
        plan.model, plan.task_name, eval_split(plan.task_name), MAIN_SCORE[plan.task_name], served_revision, fetch
    )
    return lookup, served_revision


def _official_notes(plan: QualityPlan, baseline: Mapping[str, Any], served_revision: str | None) -> list[str]:
    notes = []
    if served_revision is not None and baseline["model_revision"] != served_revision:
        notes.append(
            f"the official number is for model revision {baseline['model_revision']}, the endpoint serves {served_revision}"
        )
    if baseline["dataset_revision"] is not None and baseline["dataset_revision"] != plan.revision:
        notes.append(
            f"the official number is for dataset revision {baseline['dataset_revision']}, this run uses {plan.revision}"
        )
    return notes


def official_baseline(plan: QualityPlan, client: Any, fetch: Fetch) -> tuple[QualityPlan, dict[str, str] | None]:
    """Look the official number up for a run with no baseline; say what happened before anything is spent."""
    if plan.reference_id is not None or plan.baseline is not None:
        return plan, None
    lookup, served_revision = _official_lookup(plan, client, fetch)
    if lookup.baseline is None:
        reading = "the run reads MEASUREMENT"
        if "limit" not in plan.task_params:
            reading += " unless you pass --baseline"
        print(f"official: none, {lookup.detail}; {reading}", flush=True)
        return plan, {"status": lookup.status, "detail": lookup.detail}
    notes = _official_notes(plan, lookup.baseline, served_revision)
    print(f"official: {lookup.detail}", flush=True)
    for note in notes:
        print(f"note: {note}", flush=True)
    detail = "; ".join([lookup.detail, *notes])
    return replace(plan, baseline=lookup.baseline), {"status": lookup.status, "detail": detail}


def report_invalid(record: Mapping[str, Any]) -> None:
    if not record["validity"]["valid"]:
        for reason in record["validity"]["reasons"]:
            print(f"invalid: {reason}", file=sys.stderr)


def quality_command(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    try:
        plan = plan_quality(args)
    except KeyError as exc:
        parser.error(f"reference {args.reference} is missing key {exc}")
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
    api_key = _resolve_api_key(parser, plan.endpoint)
    data = load_quality_data(plan)
    with SIEClient(plan.endpoint, api_key=api_key, timeout_s=plan.request_timeout_s) as client:
        plan, official_lookup = official_baseline(plan, client, http_fetch)
        record, result_path = run_quality(plan, data, client, official_lookup=official_lookup)
    print(result_path)
    for line in format_rows(record["metrics"], record["comparison"]):
        print(line)
    for line in format_verdict(record["verdict"], record["baseline"], record["run"]["command"]):
        print(line)
    report_invalid(record)
    if record["run"]["predictions_path"] is None:
        raise SystemExit(1)


def performance_command(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    try:
        plan = plan_performance(args)
    except KeyError as exc:
        parser.error(f"reference {args.reference} is missing key {exc}")
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
    api_key = _resolve_api_key(parser, plan.endpoint)
    data = load_performance_data(plan)
    with SIEClient(
        plan.endpoint, api_key=api_key, timeout_s=plan.request_timeout_s, max_connections=2 * plan.max_in_flight
    ) as client:
        record, result_path = run_performance(plan, data, client)
    print(result_path)
    for line in format_rows(record["metrics"], record["comparison"]):
        print(line)
    report_invalid(record)
    if record["run"]["requests_path"] is None:
        raise SystemExit(1)


def latency_command(parser: argparse.ArgumentParser, args: argparse.Namespace, fetch: Fetch = http_fetch) -> None:
    try:
        endpoint = normalize_endpoint(args.endpoint)
    except ValueError as exc:
        parser.error(str(exc))
    api_key = _resolve_api_key(parser, endpoint)
    try:
        with SIEClient(endpoint, api_key=api_key, timeout_s=latency_mode.REQUEST_TIMEOUT_S) as client:
            record, result_path = latency_mode.run_latency(
                args.page,
                client,
                fetch,
                endpoint=_without_credentials(endpoint),
                client_description=client_description(),
                results_dir=RESULTS_DIR,
            )
    except (latency_mode.InputMismatch, NotFound, LookupFailed) as exc:
        print(f"stopped: {exc}", file=sys.stderr)
        raise SystemExit(1) from None
    print(result_path)
    for line in latency_mode.format_report(record):
        print(line)
    if record["run"]["failure_count"] == record["run"]["request_count"]:
        raise SystemExit(1)


def main(argv: Sequence[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "quality":
        quality_command(parser, args)
    elif args.command == "performance":
        performance_command(parser, args)
    elif args.command == "latency":
        latency_command(parser, args)


if __name__ == "__main__":
    main()
