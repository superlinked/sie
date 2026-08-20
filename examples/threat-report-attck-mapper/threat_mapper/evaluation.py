from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path
from typing import Any


def _mean(values: Iterable[float]) -> float:
    rows = list(values)
    return sum(rows) / len(rows) if rows else 0.0


def evaluate_predictions(predictions: list[dict[str, Any]]) -> dict[str, Any]:
    eligible = [row for row in predictions if row.get("gold_ids")]

    def ids(row: dict[str, Any], field: str) -> list[str]:
        return [str(item["technique_id"]) for item in row.get(field, [])]

    retrieval_hit_1: list[float] = []
    retrieval_hit_5: list[float] = []
    retrieval_mrr: list[float] = []
    retrieval_recall_10: list[float] = []
    per_document: dict[str, list[float]] = defaultdict(list)
    for row in eligible:
        gold = {str(value) for value in row["gold_ids"]}
        ranking = ids(row, "retrieval")
        first_rank = next((index + 1 for index, value in enumerate(ranking) if value in gold), None)
        retrieval_hit_1.append(float(bool(ranking and ranking[0] in gold)))
        retrieval_hit_5.append(float(bool(gold.intersection(ranking[:5]))))
        retrieval_mrr.append(0.0 if first_rank is None else 1.0 / first_rank)
        retrieval_recall_10.append(len(gold.intersection(ranking[:10])) / len(gold))
        per_document[str(row["document"])].append(float(bool(ranking and ranking[0] in gold)))

    reranked = [row for row in eligible if row.get("rerank")]
    rerank_hit_1: list[float] = []
    rerank_hit_5: list[float] = []
    for row in reranked:
        gold = {str(value) for value in row["gold_ids"]}
        ranking = ids(row, "rerank")
        rerank_hit_1.append(float(bool(ranking and ranking[0] in gold)))
        rerank_hit_5.append(float(bool(gold.intersection(ranking[:5]))))
    verified = [row for row in reranked if isinstance(row.get("verification"), dict)]
    covered = [row for row in verified if row["verification"].get("support") == "supported"]
    correct_covered = [row for row in covered if row["verification"].get("selected_technique_id") in row["gold_ids"]]
    return {
        "cases": {
            "total": len(predictions),
            "eligible": len(eligible),
            "excluded_no_active_gold": len(predictions) - len(eligible),
            "documents": len({row["document"] for row in eligible}),
        },
        "retrieval": {
            "hit_at_1": _mean(retrieval_hit_1),
            "hit_at_5": _mean(retrieval_hit_5),
            "recall_at_10": _mean(retrieval_recall_10),
            "mrr": _mean(retrieval_mrr),
            "document_macro_hit_at_1": _mean(_mean(values) for values in per_document.values()),
        },
        "rerank": {
            "cases": len(reranked),
            "hit_at_1": _mean(rerank_hit_1),
            "hit_at_5": _mean(rerank_hit_5),
        },
        "verification": {
            "cases": len(verified),
            "coverage": len(covered) / len(verified) if verified else 0.0,
            "selective_precision": len(correct_covered) / len(covered) if covered else 0.0,
            "abstain_or_review_rate": (len(verified) - len(covered)) / len(verified) if verified else 0.0,
        },
    }


def read_predictions(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            rows.append(json.loads(line))
    return rows
