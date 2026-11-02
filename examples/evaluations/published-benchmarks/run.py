"""Repeat the method behind the published benchmark numbers from one client process."""

import argparse
import json
import os
import platform
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import sie_sdk
from scipy.sparse import csr_matrix
from sie_sdk import SIEClient

from score import (
    HF_DATASETS,
    SCORER_ID,
    Qrels,
    compare,
    format_rows,
    load_corpus,
    load_qrels,
    load_queries,
    load_top_ranked,
    metric_rows,
    parse_task_key,
    scorer_version,
    task_metrics,
    write_json,
)

DEFAULT_ENDPOINT = "https://api.superlinked.com"
DEFAULT_TIMEOUT_S = 300.0
TOP_K = 1000
DOCUMENT_CHUNK = 500
QUERY_TOKEN_ROWS = 256
RESULTS_DIR = Path("results")
SCORE_TASKS = frozenset({"mteb/AskUbuntuDupQuestions"})
ENCODE_OUTPUTS = ("sparse", "multivector")
REFERENCE_KIND = "published-benchmark-reference"
RESULT_KIND = "published-benchmark-reproduction"

Corpus = dict[str, dict[str, str]]
Queries = dict[str, str]
TopRanked = dict[str, list[str]]
PredictionMap = dict[str, dict[str, float]]


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

    @property
    def route(self) -> str:
        return self.model if self.profile == "default" else f"{self.model}:{self.profile}"


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


@dataclass
class CallLog:
    request_count: int = 0
    failure_count: int = 0
    sdk_retries: int = 0
    execution_revisions: list[str] = field(default_factory=list)
    failure: str | None = None

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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    quality = commands.add_parser("quality", help="Repeat a published quality measurement")
    quality.add_argument("--reference", type=Path, help="A references/<id>.json file")
    quality.add_argument("--model", help="Model route, model[:profile]")
    quality.add_argument("--task", help="Task key, for example mteb/SciFact")
    quality.add_argument("--output", choices=ENCODE_OUTPUTS, help="Output contract of an encode task")
    quality.add_argument("--items-per-request", type=int, help="Items per encode request")
    quality.add_argument("--endpoint", default=DEFAULT_ENDPOINT, help="SIE endpoint URL")
    return parser


def plan_quality(args: argparse.Namespace) -> QualityPlan:
    if args.reference is not None:
        reference = json.loads(Path(args.reference).read_text(encoding="utf-8"))
        return _plan_from_reference(args, reference)
    return _plan_from_flags(args)


def _endpoint_reasons(endpoint: str) -> list[str]:
    return [] if endpoint == DEFAULT_ENDPOINT else [f"endpoint {endpoint} != {DEFAULT_ENDPOINT}"]


def _check_limit(params: Mapping[str, str]) -> None:
    if "limit" in params and (not params["limit"].isdigit() or int(params["limit"]) < 1):
        raise ValueError("limit must be a positive integer")


def _plan_from_reference(args: argparse.Namespace, reference: Mapping[str, Any]) -> QualityPlan:
    if reference.get("kind") != REFERENCE_KIND or reference.get("evaluation_mode") != "quality":
        raise ValueError("--reference must name a quality published-benchmark-reference file")
    for flag, value in (
        ("--model", args.model),
        ("--output", args.output),
        ("--items-per-request", args.items_per_request),
    ):
        if value is not None:
            raise ValueError(f"{flag} cannot be combined with --reference; the reference fixes it")
    reference_id = reference["reference_id"]
    if not reference_id or Path(reference_id).name != reference_id or reference_id in {".", ".."}:
        raise ValueError(f"reference_id {reference_id!r} is not a plain file name")
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
    processes = reference["client"]["worker_count"] * reference["client"]["processes_per_worker"]
    if processes != 1:
        reasons.append(f"client processes 1 != {processes}")
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
        endpoint=args.endpoint,
        model_version=reference["model_version"],
        comparable_reasons=reasons,
    )


def _plan_from_flags(args: argparse.Namespace) -> QualityPlan:
    if args.model is None or args.task is None:
        raise ValueError("--model and --task are required without --reference")
    task_name, params = parse_task_key(args.task)
    _check_limit(params)
    model, _, profile = args.model.partition(":")
    operation = "score" if task_name in SCORE_TASKS else "encode"
    if operation == "encode" and (args.output is None or args.items_per_request is None):
        raise ValueError("--output and --items-per-request are required for encode tasks")
    if operation == "score" and (args.output is not None or args.items_per_request is not None):
        raise ValueError("--output and --items-per-request are not accepted for the score task")
    if args.items_per_request is not None and args.items_per_request < 1:
        raise ValueError("--items-per-request must be a positive integer")
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
        endpoint=args.endpoint,
        model_version=None,
        comparable_reasons=reasons,
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
    if plan.operation == "score":
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


def normalized_rows(vectors: Any) -> np.ndarray:
    rows = np.asarray(vectors, dtype=np.float32)
    if rows.ndim != 2 or rows.shape[0] == 0:
        raise ValueError("a multivector needs at least one token row")
    norms = np.linalg.norm(rows, axis=1, keepdims=True)
    return rows / np.where(norms > 0, norms, 1.0)


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


def encode_predictions(client: Any, plan: QualityPlan, data: QualityData, log: CallLog) -> PredictionMap:
    document_ids = list(data.corpus)
    document_texts = [document_text(data.corpus[doc_id]) for doc_id in document_ids]
    documents = encode_texts(client, plan, document_ids, document_texts, log, is_query=False)
    query_ids = list(data.queries)
    queries = encode_texts(client, plan, query_ids, [data.queries[q] for q in query_ids], log, is_query=True)
    if plan.output_contract == "sparse":
        scores = sparse_scores(queries, documents)
    else:
        scores = maxsim_scores([q["multivector"] for q in queries], [d["multivector"] for d in documents])
    return top_k_predictions(scores, query_ids, document_ids)


def rerank_predictions(client: Any, plan: QualityPlan, data: QualityData, log: CallLog) -> PredictionMap:
    top_ranked = data.top_ranked or {}
    predictions: PredictionMap = {}
    for query_id, text in data.queries.items():
        candidates = top_ranked[query_id]
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
        scored = {entry["item_id"]: float(entry["score"]) for entry in result["scores"]}
        if set(scored) != set(candidates):
            raise ValueError(f"score returned item ids that differ from the candidates of query {query_id}")
        predictions[query_id] = scored
    return predictions


def client_description() -> str:
    return f"single process, {platform.platform()}, python {platform.python_version()}, sie-sdk {sie_sdk.__version__}"


def _timestamp(moment: datetime) -> str:
    return moment.strftime("%Y-%m-%dT%H:%M:%SZ")


def _slug(plan: QualityPlan) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "-", f"{plan.route}-{plan.task_key}").strip("-")


def comparison_basis(plan: QualityPlan, comparable: bool, reasons: Sequence[str]) -> str | None:
    if plan.reference_id is None:
        return None
    if comparable:
        return "same parameters"
    return "same method, different parameters: " + "; ".join(reasons)


def build_record(
    plan: QualityPlan,
    data: QualityData,
    log: CallLog,
    started: datetime,
    finished: datetime,
    predictions_path: Path | None,
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    reasons = list(plan.comparable_reasons)
    if len(log.execution_revisions) > 1:
        reasons.append("execution revision changed")
    comparable = not reasons
    validity_reasons = [f"sdk retries {log.sdk_retries}"] if log.sdk_retries else []
    if log.failure is not None:
        validity_reasons.append(f"request failed: {log.failure}")
    candidate_counts = [len(candidates) for candidates in (data.top_ranked or {}).values()]
    return {
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
        "execution_revisions": list(log.execution_revisions),
        "ingress": {"url": plan.endpoint},
        "task_key": plan.task_key,
        "dataset": {
            "name": plan.task_name,
            "revision": plan.revision,
            "subset": "test",
            "parameters": [{"name": name, "value": value} for name, value in plan.task_params.items()],
        },
        "payload": {
            "items_per_request": plan.items_per_request or max(candidate_counts, default=0),
            "output_contract": plan.output_contract,
            "parameters": [],
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
        },
        "validity": {"valid": not validity_reasons, "reasons": validity_reasons, "sdk_retries": log.sdk_retries},
        "metrics": rows,
        "comparison": compare(rows, plan.reference_metrics) if plan.reference_id is not None else [],
        "comparison_basis": comparison_basis(plan, comparable, reasons),
    }


def run_quality(
    plan: QualityPlan,
    data: QualityData,
    client: Any,
    *,
    results_dir: Path = RESULTS_DIR,
    now: Callable[[], datetime] | None = None,
) -> tuple[dict[str, Any], Path]:
    clock = now or (lambda: datetime.now(UTC))
    started = clock()
    log = CallLog()
    predictions: PredictionMap | None = None
    try:
        if plan.operation == "score":
            predictions = rerank_predictions(client, plan, data, log)
        else:
            predictions = encode_predictions(client, plan, data, log)
    except RequestFailed:
        predictions = None
    finished = clock()
    stem = f"{plan.reference_id or _slug(plan)}-{started.strftime('%Y%m%dT%H%M%SZ')}"
    results_dir.mkdir(parents=True, exist_ok=True)
    result_path = results_dir / f"{stem}.json"
    predictions_path = None
    rows: list[dict[str, Any]] = []
    if predictions is not None:
        predictions_path = results_dir / f"{stem}.predictions.json"
        write_json(predictions_path, predictions, compact=True)
        rows = metric_rows(task_metrics(predictions, data.qrels, plan.task_name))
    record = build_record(plan, data, log, started, finished, predictions_path, rows)
    write_json(result_path, record)
    return record, result_path


def quality_command(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    try:
        plan = plan_quality(args)
    except (ValueError, KeyError, OSError) as exc:
        parser.error(str(exc))
    api_key = os.environ.get("SIE_API_KEY")
    if not api_key:
        parser.error("SIE_API_KEY must be set in the environment")
    data = load_quality_data(plan)
    with SIEClient(plan.endpoint, api_key=api_key, timeout_s=plan.request_timeout_s) as client:
        record, result_path = run_quality(plan, data, client)
    print(result_path)
    for line in format_rows(record["metrics"], record["comparison"]):
        print(line)
    if record["run"]["failure_count"]:
        raise SystemExit(1)


def main(argv: Sequence[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "quality":
        quality_command(parser, args)


if __name__ == "__main__":
    main()
