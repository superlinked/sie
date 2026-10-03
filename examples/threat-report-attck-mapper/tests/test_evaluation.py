from __future__ import annotations

from threat_mapper.evaluation import evaluate_full_report_predictions, evaluate_predictions
from threat_mapper.models import GoldTechniqueMention


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


def test_full_report_evaluation_scores_span_linking_and_checks_evidence() -> None:
    report = "The actor stole a session cookie and then ran a script."
    cookie_start = report.index("stole")
    script_start = report.index("ran")
    predictions = [
        {
            "document": "report",
            "quote": "stole a session cookie",
            "source_start": cookie_start,
            "source_end": cookie_start + len("stole a session cookie"),
            "support": "supported",
            "selected_technique_id": "T1539",
            "candidates": [
                {
                    "technique_id": "T1539",
                    "dense_rank": 31,
                    "late_interaction_rank": 4,
                    "rerank_rank": 0,
                }
            ],
        },
        {
            "document": "report",
            "quote": "ran a script",
            "source_start": script_start,
            "source_end": script_start + len("ran a script"),
            "support": "supported",
            "selected_technique_id": "T9999",
            "candidates": [
                {
                    "technique_id": "T9999",
                    "dense_rank": 2,
                    "late_interaction_rank": 3,
                    "rerank_rank": 0,
                }
            ],
        },
    ]
    gold = [
        GoldTechniqueMention("implicit", "report", "T1539", "CI", "session cookie", 18, 32),
        GoldTechniqueMention("explicit", "report", "T1059", "CE", "script", 48, 54),
    ]

    metrics = evaluate_full_report_predictions(
        predictions,
        gold,
        {"report": report},
        suggestion_annotated_span_precision_gate=0.5,
        report_pair_behavior_recall_gate=1.0,
        report_pair_family_conditional_finalist_recall_gate=0.5,
    )

    assert metrics["metrics"]["implicit_recall"] == 1.0
    assert metrics["metrics"]["overall_precision"] == 0.5
    assert metrics["metrics"]["annotated_span_precision"] == 0.5
    assert metrics["predictions"]["invalid_evidence"] == 0
    assert metrics["funnel"]["implicit"] == {
        "gold_mentions": 1,
        "behavior_found": 1,
        "behavior_recall": 1.0,
        "finalist_reached": 1,
        "finalist_recall": 1.0,
        "family_finalist_reached": 1,
        "family_finalist_recall": 1.0,
        "final_matched": 1,
        "final_recall": 1.0,
        "family_final_matched": 1,
        "family_final_recall": 1.0,
    }
    assert metrics["retrieval_contribution"]["matched_mappings_by_candidate_source"] == {
        "dense_and_late_interaction": 0,
        "dense_only": 0,
        "late_interaction_only": 1,
        "candidate_ledger_unavailable": 0,
    }
    assert metrics["retrieval_contribution"]["exact_finalists_by_candidate_source"] == {
        "dense_and_late_interaction": 0,
        "dense_only": 0,
        "late_interaction_only": 1,
        "candidate_ledger_unavailable": 0,
    }
    assert metrics["retrieval_contribution"]["family_finalists_by_candidate_source"] == {
        "dense_and_late_interaction": 0,
        "dense_only": 0,
        "late_interaction_only": 1,
        "candidate_ledger_unavailable": 0,
    }
    assert metrics["gates"]["passed"] is True


def test_full_report_evaluation_reports_parent_subtechnique_family_matches_separately() -> None:
    report = "The implant captured each keystroke."
    start = report.index("captured")
    predictions = [
        {
            "document": "report",
            "quote": "captured each keystroke",
            "source_start": start,
            "source_end": len(report) - 1,
            "support": "supported",
            "selected_technique_id": "T1056.001",
            "candidates": [
                {
                    "technique_id": "T1056.001",
                    "dense_rank": 25,
                    "late_interaction_rank": 2,
                    "rerank_rank": 0,
                }
            ],
        }
    ]
    gold = [
        GoldTechniqueMention(
            "implicit",
            "report",
            "T1056",
            "CI",
            "captured each keystroke",
            start,
            len(report) - 1,
        )
    ]

    metrics = evaluate_full_report_predictions(predictions, gold, {"report": report})

    assert metrics["metrics"]["implicit_recall"] == 0.0
    assert metrics["metrics"]["implicit_family_recall"] == 1.0
    assert metrics["metrics"]["annotated_span_family_precision"] == 1.0
    assert metrics["funnel"]["implicit"]["family_finalist_recall"] == 1.0
    assert (
        metrics["retrieval_contribution"]["family_matched_mappings_by_candidate_source"]["late_interaction_only"] == 1
    )
    assert metrics["retrieval_contribution"]["family_finalists_by_candidate_source"]["late_interaction_only"] == 1
    assert metrics["family_matches"][0]["gold_technique_id"] == "T1056"
    assert metrics["family_matches"][0]["predicted_technique_id"] == "T1056.001"


def test_full_report_evaluation_collapses_repeated_mentions_at_the_agent_output_unit() -> None:
    report = "Ransomware encrypted files. The ransomware note followed."
    predictions = [
        {
            "document": "report",
            "quote": "encrypted files",
            "source_start": 11,
            "source_end": 26,
            "support": "supported",
            "selected_technique_id": "T1486",
            "candidates": [{"technique_id": "T1486", "dense_rank": 0, "late_interaction_rank": 1}],
        }
    ]
    gold = [
        GoldTechniqueMention("one", "report", "T1486", "CI", "Ransomware", 0, 10),
        GoldTechniqueMention("two", "report", "T1486", "CI", "encrypted files", 11, 26),
        GoldTechniqueMention("three", "report", "T1486", "CI", "ransomware", 32, 42),
    ]

    metrics = evaluate_full_report_predictions(predictions, gold, {"report": report})

    assert metrics["metrics"]["overall_recall"] == 1 / 3
    assert metrics["report_technique"]["exact"] == {
        "gold_pairs": 1,
        "predicted_pairs": 1,
        "correct_pairs": 1,
        "precision": 1.0,
        "recall": 1.0,
        "implicit_only_gold_pairs": 1,
        "implicit_only_correct_pairs": 1,
        "implicit_only_recall": 1.0,
        "funnel": {
            "behavior_found": 1,
            "behavior_recall": 1.0,
            "finalist_reached": 1,
            "finalist_recall": 1.0,
            "conditional_finalist_recall": 1.0,
            "final_supported": 1,
            "final_recall": 1.0,
        },
    }
