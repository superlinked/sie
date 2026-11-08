"""Recompute a result's metrics from its predictions or requests file; print the deltas and any verdict."""

import argparse
import importlib.metadata
import json
import math
from collections.abc import Mapping, Sequence
from fractions import Fraction
from pathlib import Path
from typing import Any
from urllib.parse import parse_qsl

import pytrec_eval
from datasets import Dataset, load_dataset

K_VALUES = (1, 3, 5, 10, 20, 100, 1000)
HF_DATASETS = {
    "mteb/SciFact": ("mteb/scifact", "d56462d0e63a25450459c4f213e49ffdb866f7f9"),
    "mteb/NFCorpus": ("mteb/nfcorpus", "ec0fa4fe99da2ff19ca1214b7966684033a58814"),
    "mteb/AskUbuntuDupQuestions": ("mteb/AskUbuntuDupQuestions", "c5691e3c48741d5f83b5cc8e630653d7a8cfc048"),
    "mteb/FiQA2018": ("mteb/fiqa", "27a168819829fe9bcd655c2df245fb19452e8e06"),
    "mteb/SCIDOCS": ("mteb/scidocs", "f8c2fcf00f625baaa80f62ec5bd9e1fff3b8ae88"),
    "mteb/CQADupstackPhysicsRetrieval": ("mteb/cqadupstack-physics", "79531abbd1fb92d06c6d6315a0cbbbf5bb247ea4"),
    "mteb/CosQA": ("CoIR-Retrieval/cosqa", "bc5efb7e9d437246ce393ed19d772e08e4a79535"),
    "mteb/StackOverflowQA": ("CoIR-Retrieval/stackoverflow-qa", "db8f169f3894c14a00251061f957b2063eef2bd5"),
    "mteb/LegalBenchConsumerContractsQA": (
        "mteb/legalbench_consumer_contracts_qa",
        "b23590301ec94e8087e2850b21d43d4956b1cca9",
    ),
    "mteb/MMarcoReranking": ("mteb/MMarcoReranking", "91c0f9d5cbd4cdc7ecd7039c5c2e57740cb62b00"),
    "mteb/T2Reranking": ("mteb/T2Reranking", "a34fe6bc0dff185af1228e49a0f6fb1de1565627"),
    "mteb/CMedQAv1-reranking": ("mteb/CMedQAv1-reranking", "f77f0345c9b6ac0a2447a3a71483d5b27393cb35"),
    "mteb/CMedQAv2-reranking": ("mteb/CMedQAv2-reranking", "01399b6956a4c6bf9edc5b6b90e176aeeb399686"),
    "mteb/NanoFiQA2018Retrieval": ("zeta-alpha-ai/NanoFiQA2018", "4163ba032953d5044a7a6244261413f609c14342"),
}
TASK_ALIASES = {
    "mteb/CMedQAv1Reranking": "mteb/CMedQAv1-reranking",
    "mteb/CMedQAv2Reranking": "mteb/CMedQAv2-reranking",
}
MAIN_SCORE = {
    "mteb/SciFact": "ndcg_at_10",
    "mteb/NFCorpus": "ndcg_at_10",
    "mteb/AskUbuntuDupQuestions": "map_at_1000",
    "mteb/FiQA2018": "ndcg_at_10",
    "mteb/SCIDOCS": "ndcg_at_10",
    "mteb/CQADupstackPhysicsRetrieval": "ndcg_at_10",
    "mteb/CosQA": "ndcg_at_10",
    "mteb/StackOverflowQA": "ndcg_at_10",
    "mteb/LegalBenchConsumerContractsQA": "ndcg_at_10",
    "mteb/MMarcoReranking": "map_at_1000",
    "mteb/T2Reranking": "map_at_1000",
    "mteb/CMedQAv1-reranking": "map_at_1000",
    "mteb/CMedQAv2-reranking": "map_at_1000",
    "mteb/NanoFiQA2018Retrieval": "ndcg_at_10",
}
ROLE_SPLITS = {
    "mteb/SciFact": {"corpus": "corpus", "queries": "queries"},
    "mteb/NFCorpus": {"corpus": "corpus", "queries": "queries"},
    "mteb/AskUbuntuDupQuestions": {"corpus": "test", "queries": "test"},
    "mteb/FiQA2018": {"corpus": "corpus", "queries": "queries"},
    "mteb/SCIDOCS": {"corpus": "corpus", "queries": "queries"},
    "mteb/CQADupstackPhysicsRetrieval": {"corpus": "corpus", "queries": "queries"},
    "mteb/CosQA": {"corpus": "corpus", "queries": "queries"},
    "mteb/StackOverflowQA": {"corpus": "corpus", "queries": "queries"},
    "mteb/LegalBenchConsumerContractsQA": {"corpus": "corpus", "queries": "queries"},
    "mteb/MMarcoReranking": {"corpus": "dev", "queries": "dev"},
    "mteb/T2Reranking": {"corpus": "dev", "queries": "dev"},
    "mteb/CMedQAv1-reranking": {"corpus": "test", "queries": "test"},
    "mteb/CMedQAv2-reranking": {"corpus": "test", "queries": "test"},
    "mteb/NanoFiQA2018Retrieval": {"corpus": "train", "queries": "train"},
}
EVAL_SPLITS = {
    "mteb/MMarcoReranking": "dev",
    "mteb/T2Reranking": "dev",
    "mteb/NanoFiQA2018Retrieval": "train",
}
DEFAULT_EVAL_SPLIT = "test"
QRELS_CONFIGS = {"mteb/NanoFiQA2018Retrieval": "qrels"}
DEFAULT_QRELS_CONFIG = "default"
IGNORE_IDENTICAL_IDS = frozenset({"mteb/FiQA2018"})
METRIC_UNIT = "ratio"
AGGREGATION = "task_native"
SCORER_ID = "pytrec-eval"
SCORER_DISTRIBUTION = "pytrec-eval-terrier"
PERFORMANCE_MODE = "isolated_performance"
PERFORMANCE_METRICS = {
    "offered_requests_per_second": ("requests_per_second", "campaign_global"),
    "achieved_requests_per_second": ("requests_per_second", "common_window"),
    "eventual_achieved_requests_per_second": ("requests_per_second", "including_drain"),
    "in_window_success_rate": ("ratio", "campaign_global"),
    "in_window_retry_free_success_rate": ("ratio", "campaign_global"),
    "success_rate": ("ratio", "campaign_global"),
    "retry_free_success_rate": ("ratio", "campaign_global"),
    "scheduling_lag_p99_ms": ("milliseconds", "p99"),
    "latency_p50_ms": ("milliseconds", "p50"),
    "latency_p95_ms": ("milliseconds", "p95"),
    "latency_p99_ms": ("milliseconds", "p99"),
}
LATENCY_PERCENTILES = (50, 95, 99)
NS_PER_S = 1_000_000_000
NS_PER_MS = 1_000_000
PRIMARY_METRIC = "main_score"
BASELINE_METRICS = frozenset(
    {
        PRIMARY_METRIC,
        "accuracy",
        *(f"{family}_at_{k}" for family in ("ndcg", "map", "recall", "precision", "mrr") for k in K_VALUES),
    }
)
DEFAULT_MARGIN = 0.98
VERDICT_PASS = "PASS"
VERDICT_FAIL_REGRESSION = "FAIL (regression)"
VERDICT_FAIL_NO_OUTPUT = "FAIL (no-output)"
VERDICT_WARN_IMPROVEMENT = "WARN (suspicious-improvement)"
VERDICT_MEASUREMENT = "MEASUREMENT"

Qrels = dict[str, dict[str, int]]
Predictions = Mapping[str, Mapping[str, float]]
Rows = list[dict[str, Any]]


def parse_task_key(task_key: str) -> tuple[str, dict[str, str]]:
    task_name, _, query = task_key.partition("?")
    task_name = TASK_ALIASES.get(task_name, task_name)
    if task_name not in HF_DATASETS:
        raise ValueError(f"unknown task {task_name!r}; known tasks: {', '.join(HF_DATASETS)}")
    pairs = parse_qsl(query, keep_blank_values=True)
    params = dict(pairs)
    if len(params) != len(pairs):
        raise ValueError(f"the task key {task_key!r} repeats a parameter")
    return task_name, params


def eval_split(task_name: str) -> str:
    return EVAL_SPLITS.get(task_name, DEFAULT_EVAL_SPLIT)


def scorer_version() -> str:
    return importlib.metadata.version(SCORER_DISTRIBUTION)


def _rows(task_name: str, config: str, split: str, revision: str) -> Dataset:
    rows = load_dataset(HF_DATASETS[task_name][0], config, split=split, revision=revision)
    if not isinstance(rows, Dataset):
        raise TypeError(f"{task_name} {config}/{split} did not load as a single split")
    return rows


def load_corpus(task_name: str, revision: str) -> dict[str, dict[str, str]]:
    rows = _rows(task_name, "corpus", ROLE_SPLITS[task_name]["corpus"], revision)
    titles = rows["title"] if "title" in rows.column_names else [""] * len(rows)
    return {
        str(doc_id): {"title": title or "", "text": text or ""}
        for doc_id, title, text in zip(rows["_id"], titles, rows["text"], strict=True)
    }


def load_queries(task_name: str, revision: str) -> dict[str, str]:
    rows = _rows(task_name, "queries", ROLE_SPLITS[task_name]["queries"], revision)
    return {str(query_id): text for query_id, text in zip(rows["_id"], rows["text"], strict=True)}


def load_qrels(task_name: str, revision: str, split: str | None = None) -> Qrels:
    config = QRELS_CONFIGS.get(task_name, DEFAULT_QRELS_CONFIG)
    rows = _rows(task_name, config, split or eval_split(task_name), revision)
    grades = rows["score"] if "score" in rows.column_names else [1] * len(rows)
    qrels: Qrels = {}
    for query_id, doc_id, grade in zip(rows["query-id"], rows["corpus-id"], grades, strict=True):
        qrels.setdefault(str(query_id), {})[str(doc_id)] = int(grade)
    return qrels


def load_top_ranked(task_name: str, revision: str, split: str | None = None) -> dict[str, list[str]]:
    rows = _rows(task_name, "top_ranked", split or eval_split(task_name), revision)
    return {
        str(query_id): [str(doc_id) for doc_id in doc_ids]
        for query_id, doc_ids in zip(rows["query-id"], rows["corpus-ids"], strict=True)
    }


def _per_query(predictions: Predictions, qrels: Qrels) -> dict[str, dict[str, float]]:
    ks = ",".join(str(k) for k in K_VALUES)
    measures = {f"map_cut.{ks}", f"ndcg_cut.{ks}", f"recall.{ks}", f"P.{ks}"}
    return pytrec_eval.RelevanceEvaluator(qrels, measures).evaluate(predictions)


def _mrr(predictions: Predictions, qrels: Qrels) -> dict[str, float]:
    k_max = max(K_VALUES)
    reciprocal_ranks: dict[int, list[float]] = {k: [] for k in K_VALUES}
    for query_id, doc_scores in predictions.items():
        hits = sorted(doc_scores.items(), key=lambda item: item[1], reverse=True)[:k_max]
        relevant = {doc_id for doc_id, grade in qrels[query_id].items() if grade > 0}
        for k in K_VALUES:
            reciprocal_rank = 0.0
            for rank, (doc_id, _) in enumerate(hits[:k]):
                if doc_id in relevant:
                    reciprocal_rank = 1.0 / (rank + 1)
                    break
            reciprocal_ranks[k].append(reciprocal_rank)
    return {f"mrr_at_{k}": sum(ranks) / len(ranks) for k, ranks in reciprocal_ranks.items()}


def retrieval_metrics(predictions: Predictions, qrels: Qrels) -> dict[str, float]:
    if set(predictions) != set(qrels):
        raise ValueError("predictions must cover exactly the qrels queries")
    per_query = _per_query(predictions, qrels)
    count = len(per_query)
    metrics: dict[str, float] = {}
    for prefix, measure in (("ndcg", "ndcg_cut"), ("map", "map_cut"), ("recall", "recall"), ("precision", "P")):
        for k in K_VALUES:
            metrics[f"{prefix}_at_{k}"] = round(sum(s[f"{measure}_{k}"] for s in per_query.values()) / count, 5)
    metrics["accuracy"] = metrics["recall_at_1"]
    metrics.update(_mrr(predictions, qrels))
    return metrics


def task_metrics(predictions: Predictions, qrels: Qrels, task_name: str) -> dict[str, float]:
    if task_name in IGNORE_IDENTICAL_IDS:
        predictions = {
            query_id: {doc_id: value for doc_id, value in doc_scores.items() if doc_id != query_id}
            for query_id, doc_scores in predictions.items()
        }
    metrics = retrieval_metrics(predictions, qrels)
    metrics["main_score"] = metrics[MAIN_SCORE[task_name]]
    return metrics


def metric_rows(metrics: Mapping[str, float]) -> Rows:
    return [
        {"name": name, "unit": METRIC_UNIT, "aggregation": AGGREGATION, "value": value}
        for name, value in sorted(metrics.items())
    ]


def nanoseconds(seconds: float | Fraction) -> int:
    return int(Fraction(str(seconds)) * NS_PER_S)


def nearest_rank(values: Sequence[float], percent: int) -> float:
    if not values:
        raise ValueError("a percentile needs at least one value")
    ordered = sorted(values)
    return ordered[max(0, math.ceil(Fraction(percent * len(ordered), 100)) - 1)]


def _retry_free(row: Mapping[str, Any]) -> bool:
    return row["attempt"] == 1 and row["client_retry_count"] == 0


def performance_metrics(rows: Sequence[Mapping[str, Any]], rate: float, duration_s: float) -> dict[str, float]:
    count = len(rows)
    if count == 0:
        raise ValueError("performance metrics need at least one scheduled request")
    window_ns = nanoseconds(duration_s)
    successes = [row for row in rows if row["status"] == "success"]
    in_window = [row for row in successes if row["completed_offset_ns"] < window_ns]
    metrics = {
        "offered_requests_per_second": float(rate),
        "achieved_requests_per_second": len(in_window) / duration_s,
        "eventual_achieved_requests_per_second": len(successes) / duration_s,
        "in_window_success_rate": len(in_window) / count,
        "in_window_retry_free_success_rate": sum(_retry_free(row) for row in in_window) / count,
        "success_rate": len(successes) / count,
        "retry_free_success_rate": sum(_retry_free(row) for row in successes) / count,
        "scheduling_lag_p99_ms": nearest_rank([row["scheduling_lag_ms"] for row in rows], 99),
    }
    if successes:
        latencies = [(row["completed_offset_ns"] - row["actual_send_offset_ns"]) / NS_PER_MS for row in successes]
        for percent in LATENCY_PERCENTILES:
            metrics[f"latency_p{percent}_ms"] = nearest_rank(latencies, percent)
    return metrics


def performance_metric_rows(metrics: Mapping[str, float]) -> Rows:
    return [
        {"name": name, "unit": unit, "aggregation": aggregation, "value": metrics[name]}
        for name, (unit, aggregation) in PERFORMANCE_METRICS.items()
        if name in metrics
    ]


def compare(rows: Sequence[Mapping[str, Any]], reference_rows: Sequence[Mapping[str, Any]]) -> Rows:
    reference = {(row["name"], row["aggregation"]): row["value"] for row in reference_rows}
    comparison: Rows = []
    for row in rows:
        key = (row["name"], row["aggregation"])
        if key not in reference:
            continue
        comparison.append(
            {
                "name": row["name"],
                "aggregation": row["aggregation"],
                "reference_value": reference[key],
                "value": row["value"],
                "delta": round(row["value"] - reference[key], 5),
            }
        )
    return comparison


def format_rows(rows: Sequence[Mapping[str, Any]], comparison: Sequence[Mapping[str, Any]]) -> list[str]:
    matched = {(row["name"], row["aggregation"]): row for row in comparison}
    width = max([20, *(len(row["name"]) for row in rows)])
    lines = []
    for row in rows:
        line = f"{row['name']:<{width}} {row['value']:.5f}"
        entry = matched.get((row["name"], row["aggregation"]))
        if entry is not None:
            line += f"  reference {entry['reference_value']:.5f}  delta {entry['delta']:+.5f}"
        lines.append(line)
    return lines


def compute_verdict(
    metrics: Mapping[str, float],
    baseline: Mapping[str, Any] | None,
    *,
    limited: bool,
    margin: float = DEFAULT_MARGIN,
) -> dict[str, Any]:
    out: dict[str, Any] = {
        "metric": PRIMARY_METRIC if baseline is None else baseline["metric"],
        "current": None,
        "baseline": None,
        "delta_pct": None,
        "threshold_low": None,
        "threshold_high": None,
        "margin": margin,
    }
    compared = None if limited else baseline
    if compared is not None:
        out["baseline"] = compared["value"]
        out["threshold_low"] = compared["value"] * margin
        out["threshold_high"] = compared["value"] / margin
    if not metrics:
        return {"verdict": VERDICT_FAIL_NO_OUTPUT, "reason": "no_predictions", **out}
    current = out["current"] = metrics[out["metric"]]
    if compared is not None:
        out["delta_pct"] = (current - compared["value"]) / compared["value"] * 100.0
    if current == 0.0:
        return {"verdict": VERDICT_FAIL_NO_OUTPUT, "reason": "current_zero", **out}
    if baseline is None:
        return {"verdict": VERDICT_MEASUREMENT, "reason": "no_baseline", **out}
    if limited:
        return {"verdict": VERDICT_MEASUREMENT, "reason": "limited_run", **out}
    if current < out["threshold_low"]:
        return {"verdict": VERDICT_FAIL_REGRESSION, "reason": "below_margin", **out}
    if current > out["threshold_high"]:
        return {"verdict": VERDICT_WARN_IMPROVEMENT, "reason": "above_margin", **out}
    return {"verdict": VERDICT_PASS, "reason": "within_margin", **out}


def format_verdict(verdict: Mapping[str, Any], baseline: Mapping[str, Any] | None, command: str) -> list[str]:
    label, reason, metric = verdict["verdict"], verdict["reason"], verdict["metric"]
    if reason == "no_baseline":
        return [f"verdict: {label} no baseline"]
    if reason == "limited_run":
        return [f"verdict: {label} limited run, not compared with a full-dataset number"]
    current, value = verdict["current"], verdict["baseline"]
    band = f"{verdict['threshold_low']:.5f} to {verdict['threshold_high']:.5f}" if value is not None else ""
    source = baseline["source"] if baseline is not None else ""
    if reason == "no_predictions":
        lines = [f"verdict: {label} no predictions"]
    elif reason == "current_zero":
        lines = [f"verdict: {label} {metric} is exactly 0"]
    else:
        delta = f"{verdict['delta_pct']:+.2f}%"
        lines = [f"verdict: {label} {metric} {current:.5f} against {value:.5f} from {source} ({delta}, band {band})"]
    if label in {VERDICT_FAIL_REGRESSION, VERDICT_FAIL_NO_OUTPUT}:
        expected = f"{metric} above 0" if value is None else f"{metric} from {band}, {value:.5f} from {source}"
        got = "no predictions" if current is None else f"{metric} {current:.5f}"
        lines += [f"expected: {expected}", f"got: {got}", f"reproduce: {command}"]
    return lines


def recompute_verdict(record: Mapping[str, Any], rows: Sequence[Mapping[str, Any]]) -> dict[str, Any] | None:
    if "baseline" not in record:
        return None
    _, params = parse_task_key(record["task_key"])
    return compute_verdict({row["name"]: row["value"] for row in rows}, record["baseline"], limited="limit" in params)


def scoped_qrels(qrels: Qrels, predictions: Predictions, params: Mapping[str, str]) -> Qrels:
    if "limit" not in params:
        return qrels
    return {query_id: qrels[query_id] for query_id in predictions if query_id in qrels}


def write_json(path: Path, payload: Any, *, compact: bool = False) -> None:
    text = json.dumps(payload, separators=(",", ":")) if compact else json.dumps(payload, indent=2)
    path.write_text(text + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.write_text("".join(json.dumps(row, separators=(",", ":")) + "\n" for row in rows), encoding="utf-8")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _sibling(result_path: Path, record: Mapping[str, Any], key: str, label: str) -> Path:
    name = record["run"][key]
    if name is None:
        raise ValueError(f"{result_path} has no {label} file; its run did not complete")
    recorded = Path(name)
    for candidate in (recorded, result_path.parent / recorded.name):
        if candidate.is_file():
            return candidate
    message = f"{result_path} names the {label} file {recorded.name}, found neither at {recorded} nor beside the result"
    if key == "predictions_path":
        message += "; the quality run has to be repeated to re-derive its metrics"
    raise ValueError(message)


def _quality_rows(result_path: Path, record: Mapping[str, Any]) -> Rows:
    predictions_path = _sibling(result_path, record, "predictions_path", "predictions")
    predictions = json.loads(predictions_path.read_text(encoding="utf-8"))
    task_name, params = parse_task_key(record["task_key"])
    dataset = record["dataset"]
    qrels = scoped_qrels(load_qrels(task_name, dataset["revision"], dataset["subset"]), predictions, params)
    return metric_rows(task_metrics(predictions, qrels, task_name))


def _performance_rows(result_path: Path, record: Mapping[str, Any]) -> Rows:
    requests = read_jsonl(_sibling(result_path, record, "requests_path", "requests"))
    level = record["schedule"]["levels"][0]
    return performance_metric_rows(performance_metrics(requests, level["rate"], level["duration_s"]))


def rescore(result_path: Path) -> tuple[dict[str, Any], Rows, Rows]:
    record = json.loads(result_path.read_text(encoding="utf-8"))
    if record.get("evaluation_mode") == PERFORMANCE_MODE:
        rows = _performance_rows(result_path, record)
    else:
        rows = _quality_rows(result_path, record)
    reference_rows = [
        {"name": row["name"], "aggregation": row["aggregation"], "value": row["reference_value"]}
        for row in record.get("comparison", [])
    ]
    return record, rows, compare(rows, reference_rows)


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result", type=Path, help="A results/<id>-<utc>.json file written by run.py")
    parser.add_argument(
        "--write", action="store_true", help="Update metrics, comparison and verdict in the result file"
    )
    args = parser.parse_args(argv)
    try:
        record, rows, comparison = rescore(args.result)
    except ValueError as exc:
        parser.error(str(exc))
    for line in format_rows(rows, comparison):
        print(line)
    verdict = recompute_verdict(record, rows)
    if verdict is not None:
        for line in format_verdict(verdict, record["baseline"], record["run"]["command"]):
            print(line)
    if args.write:
        record["metrics"] = rows
        record["comparison"] = comparison
        if verdict is not None:
            record["verdict"] = verdict
        write_json(args.result, record)


if __name__ == "__main__":
    main()
