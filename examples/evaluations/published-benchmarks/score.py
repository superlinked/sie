"""Recompute the published quality metrics from a result's predictions file and print the deltas."""

import argparse
import importlib.metadata
import json
from collections.abc import Mapping, Sequence
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
}
MAIN_SCORE = {
    "mteb/SciFact": "ndcg_at_10",
    "mteb/NFCorpus": "ndcg_at_10",
    "mteb/AskUbuntuDupQuestions": "map_at_1000",
}
ROLE_SPLITS = {
    "mteb/SciFact": {"corpus": "corpus", "queries": "queries"},
    "mteb/NFCorpus": {"corpus": "corpus", "queries": "queries"},
    "mteb/AskUbuntuDupQuestions": {"corpus": "test", "queries": "test"},
}
METRIC_UNIT = "ratio"
AGGREGATION = "task_native"
SCORER_ID = "pytrec-eval"
SCORER_DISTRIBUTION = "pytrec-eval-terrier"

Qrels = dict[str, dict[str, int]]
Predictions = Mapping[str, Mapping[str, float]]
Rows = list[dict[str, Any]]


def parse_task_key(task_key: str) -> tuple[str, dict[str, str]]:
    task_name, _, query = task_key.partition("?")
    if task_name not in HF_DATASETS:
        raise ValueError(f"unknown task {task_name!r}; known tasks: {', '.join(HF_DATASETS)}")
    return task_name, dict(parse_qsl(query, keep_blank_values=True))


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


def load_qrels(task_name: str, revision: str, split: str = "test") -> Qrels:
    rows = _rows(task_name, "default", split, revision)
    qrels: Qrels = {}
    for query_id, doc_id, grade in zip(rows["query-id"], rows["corpus-id"], rows["score"], strict=True):
        qrels.setdefault(str(query_id), {})[str(doc_id)] = int(grade)
    return qrels


def load_top_ranked(task_name: str, revision: str, split: str = "test") -> dict[str, list[str]]:
    rows = _rows(task_name, "top_ranked", split, revision)
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
    metrics = retrieval_metrics(predictions, qrels)
    metrics["main_score"] = metrics[MAIN_SCORE[task_name]]
    return metrics


def metric_rows(metrics: Mapping[str, float]) -> Rows:
    return [
        {"name": name, "unit": METRIC_UNIT, "aggregation": AGGREGATION, "value": value}
        for name, value in sorted(metrics.items())
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
    lines = []
    for row in rows:
        line = f"{row['name']:<20} {row['value']:.5f}"
        entry = matched.get((row["name"], row["aggregation"]))
        if entry is not None:
            line += f"  reference {entry['reference_value']:.5f}  delta {entry['delta']:+.5f}"
        lines.append(line)
    return lines


def scoped_qrels(qrels: Qrels, predictions: Predictions, params: Mapping[str, str]) -> Qrels:
    if "limit" not in params:
        return qrels
    return {query_id: qrels[query_id] for query_id in predictions if query_id in qrels}


def write_json(path: Path, payload: Any, *, compact: bool = False) -> None:
    text = json.dumps(payload, separators=(",", ":")) if compact else json.dumps(payload, indent=2)
    path.write_text(text + "\n", encoding="utf-8")


def rescore(result_path: Path) -> tuple[dict[str, Any], Rows, Rows]:
    record = json.loads(result_path.read_text(encoding="utf-8"))
    predictions_name = record["run"]["predictions_path"]
    if predictions_name is None:
        raise ValueError(f"{result_path} has no predictions file; its run did not complete")
    predictions_path = result_path.parent / Path(predictions_name).name
    predictions = json.loads(predictions_path.read_text(encoding="utf-8"))
    task_name, params = parse_task_key(record["task_key"])
    dataset = record["dataset"]
    qrels = scoped_qrels(load_qrels(task_name, dataset["revision"], dataset["subset"]), predictions, params)
    rows = metric_rows(task_metrics(predictions, qrels, task_name))
    reference_rows = [
        {"name": row["name"], "aggregation": row["aggregation"], "value": row["reference_value"]}
        for row in record.get("comparison", [])
    ]
    return record, rows, compare(rows, reference_rows)


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result", type=Path, help="A results/<id>-<utc>.json file written by run.py")
    parser.add_argument("--write", action="store_true", help="Update metrics and comparison in the result file")
    args = parser.parse_args(argv)
    try:
        record, rows, comparison = rescore(args.result)
    except ValueError as exc:
        parser.error(str(exc))
    for line in format_rows(rows, comparison):
        print(line)
    if args.write:
        record["metrics"] = rows
        record["comparison"] = comparison
        write_json(args.result, record)


if __name__ == "__main__":
    main()
