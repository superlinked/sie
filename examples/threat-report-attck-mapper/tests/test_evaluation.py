from __future__ import annotations

from threat_mapper.evaluation import evaluate_predictions


def candidate(technique_id: str) -> dict[str, object]:
    return {"technique_id": technique_id, "name": technique_id, "dense_score": 0.5}


def test_evaluation_reports_retrieval_rerank_and_selective_verifier_metrics() -> None:
    rows = [
        {
            "document": "one",
            "gold_ids": ["T1539"],
            "retrieval": [candidate("T1539"), candidate("T1557")],
            "rerank": [candidate("T1557"), candidate("T1539")],
            "verification": {"support": "supported", "selected_technique_id": "T1539"},
        },
        {
            "document": "two",
            "gold_ids": ["T1557", "T1539"],
            "retrieval": [candidate("T1000"), candidate("T1557")],
            "rerank": [candidate("T1557"), candidate("T1000")],
            "verification": {"support": "ambiguous", "selected_technique_id": "T1557"},
        },
    ]

    metrics = evaluate_predictions(rows)

    assert metrics["retrieval"]["hit_at_1"] == 0.5
    assert metrics["retrieval"]["hit_at_5"] == 1.0
    assert metrics["rerank"]["hit_at_1"] == 0.5
    assert metrics["verification"]["coverage"] == 0.5
    assert metrics["verification"]["selective_precision"] == 1.0


def test_evaluation_returns_zero_metrics_when_no_cases_have_active_gold() -> None:
    metrics = evaluate_predictions(
        [
            {
                "document": "historical-report",
                "gold_ids": [],
                "retrieval": [candidate("T1539")],
                "rerank": [],
                "verification": None,
            }
        ]
    )

    assert metrics["cases"] == {
        "total": 1,
        "eligible": 0,
        "excluded_no_active_gold": 1,
        "documents": 0,
    }
    assert metrics["retrieval"] == {
        "hit_at_1": 0.0,
        "hit_at_5": 0.0,
        "recall_at_10": 0.0,
        "mrr": 0.0,
        "document_macro_hit_at_1": 0.0,
    }
