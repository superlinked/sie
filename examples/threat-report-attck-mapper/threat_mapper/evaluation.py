from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from .models import GoldTechniqueMention


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


def _overlap(left_start: int, left_end: int, right_start: int, right_end: int) -> int:
    return max(0, min(left_end, right_end) - max(left_start, right_start))


def _technique_family(technique_id: str) -> str:
    return technique_id.split(".", 1)[0]


def evaluate_full_report_predictions(
    predictions: list[dict[str, Any]],
    gold_mentions: list[GoldTechniqueMention],
    reports: dict[str, str],
    *,
    excluded_documents: set[str] | None = None,
    suggestion_annotated_span_precision_gate: float = 0.85,
    report_pair_behavior_recall_gate: float = 0.70,
    report_pair_family_conditional_finalist_recall_gate: float = 0.90,
    dense_pool_count: int = 20,
    late_interaction_pool_count: int = 20,
    exemplar_pool_count: int = 0,
) -> dict[str, Any]:
    excluded = excluded_documents or set()

    def is_suggested(row: dict[str, Any]) -> bool:
        route = row.get("route")
        if route is None:
            return row.get("support") == "supported"
        return route == "suggested_mapping"

    def candidate_source(candidate: dict[str, Any]) -> str:
        dense_rank = candidate.get("dense_rank")
        late_rank = candidate.get("late_interaction_rank")
        exemplar_rank = candidate.get("exemplar_rank")
        sources = []
        if type(dense_rank) is int and dense_rank < dense_pool_count:
            sources.append("dense")
        if type(late_rank) is int and late_rank < late_interaction_pool_count:
            sources.append("late_interaction")
        if type(exemplar_rank) is int and exemplar_rank < exemplar_pool_count:
            sources.append("exemplar")
        if not sources:
            return "candidate_ledger_unavailable"
        if len(sources) == 1 and sources[0] != "exemplar":
            return f"{sources[0]}_only"
        if sources == ["exemplar"]:
            return "exemplar_only"
        return "_and_".join(sources)

    invalid_evidence: list[dict[str, Any]] = []
    for row in predictions:
        document = str(row.get("document", ""))
        report = reports.get(document, "")
        start = row.get("source_start")
        end = row.get("source_end")
        quote = str(row.get("quote", ""))
        valid_span = type(start) is int and type(end) is int and 0 <= start < end <= len(report)
        if not valid_span or report[start:end] != quote:
            invalid_evidence.append({"document": document, "source_start": start, "source_end": end, "quote": quote})

    supported_by_key: dict[tuple[str, str, int, int], dict[str, Any]] = {}
    for row in predictions:
        technique_id = row.get("selected_technique_id")
        if not is_suggested(row) or not isinstance(technique_id, str) or not technique_id:
            continue
        key = (
            str(row.get("document", "")),
            technique_id,
            int(row.get("source_start", -1)),
            int(row.get("source_end", -1)),
        )
        supported_by_key[key] = row
    supported = list(supported_by_key.values())

    gold_by_match: dict[tuple[str, str], list[GoldTechniqueMention]] = defaultdict(list)
    for gold in gold_mentions:
        gold_by_match[(gold.document, gold.technique_id)].append(gold)
    matched_gold: set[str] = set()
    matches: list[dict[str, Any]] = []
    false_positives: list[dict[str, Any]] = []
    for row in sorted(
        supported,
        key=lambda item: (str(item.get("document", "")), int(item.get("source_start", -1))),
    ):
        candidates = [
            gold
            for gold in gold_by_match[(str(row["document"]), str(row["selected_technique_id"]))]
            if gold.mention_id not in matched_gold
            and _overlap(
                int(row["source_start"]),
                int(row["source_end"]),
                gold.source_start,
                gold.source_end,
            )
            > 0
        ]
        if not candidates:
            false_positives.append(row)
            continue
        match = max(
            candidates,
            key=lambda gold: _overlap(
                int(row["source_start"]), int(row["source_end"]), gold.source_start, gold.source_end
            ),
        )
        matched_gold.add(match.mention_id)
        selected_candidate = next(
            (
                candidate
                for candidate in row.get("candidates", [])
                if candidate.get("technique_id") == match.technique_id
            ),
            {},
        )
        dense_rank = selected_candidate.get("dense_rank")
        late_rank = selected_candidate.get("late_interaction_rank")
        exemplar_rank = selected_candidate.get("exemplar_rank")
        retrieval_path = candidate_source(selected_candidate)
        matches.append(
            {
                "document": match.document,
                "technique_id": match.technique_id,
                "annotation_class": match.annotation_class,
                "gold_quote": match.quote,
                "predicted_quote": row["quote"],
                "source_start": row["source_start"],
                "source_end": row["source_end"],
                "retrieval_path": retrieval_path,
                "dense_rank": dense_rank,
                "late_interaction_rank": late_rank,
                "exemplar_rank": exemplar_rank,
                "rerank_rank": selected_candidate.get("rerank_rank"),
            }
        )

    misses = [gold.to_dict() for gold in gold_mentions if gold.mention_id not in matched_gold]

    family_gold_by_match: dict[tuple[str, str], list[GoldTechniqueMention]] = defaultdict(list)
    for gold in gold_mentions:
        family_gold_by_match[(gold.document, _technique_family(gold.technique_id))].append(gold)
    family_matched_gold: set[str] = set()
    family_matches: list[dict[str, Any]] = []
    family_false_positives: list[dict[str, Any]] = []
    for row in sorted(
        supported,
        key=lambda item: (str(item.get("document", "")), int(item.get("source_start", -1))),
    ):
        predicted_id = str(row["selected_technique_id"])
        candidates = [
            gold
            for gold in family_gold_by_match[(str(row["document"]), _technique_family(predicted_id))]
            if gold.mention_id not in family_matched_gold
            and _overlap(
                int(row["source_start"]),
                int(row["source_end"]),
                gold.source_start,
                gold.source_end,
            )
            > 0
        ]
        if not candidates:
            family_false_positives.append(row)
            continue
        match = max(
            candidates,
            key=lambda gold: _overlap(
                int(row["source_start"]), int(row["source_end"]), gold.source_start, gold.source_end
            ),
        )
        family_matched_gold.add(match.mention_id)
        selected_candidate = next(
            (candidate for candidate in row.get("candidates", []) if candidate.get("technique_id") == predicted_id),
            {},
        )
        dense_rank = selected_candidate.get("dense_rank")
        late_rank = selected_candidate.get("late_interaction_rank")
        exemplar_rank = selected_candidate.get("exemplar_rank")
        retrieval_path = candidate_source(selected_candidate)
        family_matches.append(
            {
                "document": match.document,
                "gold_technique_id": match.technique_id,
                "predicted_technique_id": predicted_id,
                "technique_family": _technique_family(match.technique_id),
                "annotation_class": match.annotation_class,
                "gold_quote": match.quote,
                "predicted_quote": row["quote"],
                "source_start": row["source_start"],
                "source_end": row["source_end"],
                "retrieval_path": retrieval_path,
                "dense_rank": dense_rank,
                "late_interaction_rank": late_rank,
                "exemplar_rank": exemplar_rank,
                "rerank_rank": selected_candidate.get("rerank_rank"),
            }
        )

    implicit = [gold for gold in gold_mentions if gold.annotation_class == "CI"]
    explicit = [gold for gold in gold_mentions if gold.annotation_class == "CE"]
    matched_implicit = sum(gold.mention_id in matched_gold for gold in implicit)
    matched_explicit = sum(gold.mention_id in matched_gold for gold in explicit)
    family_matched_implicit = sum(gold.mention_id in family_matched_gold for gold in implicit)
    family_matched_explicit = sum(gold.mention_id in family_matched_gold for gold in explicit)
    supported_on_annotated_spans = [
        row
        for row in supported
        if any(
            gold.document == row.get("document")
            and _overlap(
                int(row["source_start"]),
                int(row["source_end"]),
                gold.source_start,
                gold.source_end,
            )
            > 0
            for gold in gold_mentions
        )
    ]
    precision = len(matches) / len(supported) if supported else 0.0
    family_precision = len(family_matches) / len(supported) if supported else 0.0
    annotated_span_precision = len(matches) / len(supported_on_annotated_spans) if supported_on_annotated_spans else 0.0
    annotated_span_family_precision = (
        len(family_matches) / len(supported_on_annotated_spans) if supported_on_annotated_spans else 0.0
    )
    implicit_recall = matched_implicit / len(implicit) if implicit else 0.0
    implicit_family_recall = family_matched_implicit / len(implicit) if implicit else 0.0
    excluded_present = sorted({str(row.get("document", "")) for row in predictions}.intersection(excluded))

    def funnel(gold_scope: list[GoldTechniqueMention]) -> dict[str, Any]:
        behavior_found = 0
        finalist_reached = 0
        family_finalist_reached_count = 0
        final_matched = 0
        family_final_matched = 0
        for gold in gold_scope:
            overlapping = [
                row
                for row in predictions
                if row.get("document") == gold.document
                and type(row.get("source_start")) is int
                and type(row.get("source_end")) is int
                and _overlap(
                    int(row["source_start"]),
                    int(row["source_end"]),
                    gold.source_start,
                    gold.source_end,
                )
                > 0
            ]
            if overlapping:
                behavior_found += 1
            if any(
                any(candidate.get("technique_id") == gold.technique_id for candidate in row.get("candidates", []))
                for row in overlapping
            ):
                finalist_reached += 1
            family_finalist_reached = any(
                any(
                    _technique_family(str(candidate.get("technique_id", ""))) == _technique_family(gold.technique_id)
                    for candidate in row.get("candidates", [])
                )
                for row in overlapping
            )
            if family_finalist_reached:
                family_finalist_reached_count += 1
            if gold.mention_id in matched_gold:
                final_matched += 1
            if gold.mention_id in family_matched_gold:
                family_final_matched += 1
        total = len(gold_scope)
        return {
            "gold_mentions": total,
            "behavior_found": behavior_found,
            "behavior_recall": behavior_found / total if total else 0.0,
            "finalist_reached": finalist_reached,
            "finalist_recall": finalist_reached / total if total else 0.0,
            "family_finalist_reached": family_finalist_reached_count,
            "family_finalist_recall": family_finalist_reached_count / total if total else 0.0,
            "final_matched": final_matched,
            "final_recall": final_matched / total if total else 0.0,
            "family_final_matched": family_final_matched,
            "family_final_recall": family_final_matched / total if total else 0.0,
        }

    def report_technique_metrics(*, family_level: bool) -> dict[str, Any]:
        gold_pairs: dict[tuple[str, str], list[GoldTechniqueMention]] = defaultdict(list)
        for gold in gold_mentions:
            technique_id = _technique_family(gold.technique_id) if family_level else gold.technique_id
            gold_pairs[(gold.document, technique_id)].append(gold)

        predicted_pairs: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
        for row in supported:
            selected_id = str(row["selected_technique_id"])
            technique_id = _technique_family(selected_id) if family_level else selected_id
            predicted_pairs[(str(row["document"]), technique_id)].append(row)

        correct_pairs = {
            pair
            for pair, rows in predicted_pairs.items()
            if pair in gold_pairs
            and any(
                _overlap(
                    int(row["source_start"]),
                    int(row["source_end"]),
                    gold.source_start,
                    gold.source_end,
                )
                > 0
                for row in rows
                for gold in gold_pairs[pair]
            )
        }
        implicit_only_pairs = {
            pair
            for pair, mentions in gold_pairs.items()
            if all(mention.annotation_class == "CI" for mention in mentions)
        }

        behavior_pairs: set[tuple[str, str]] = set()
        finalist_pairs: set[tuple[str, str]] = set()
        for pair, mentions in gold_pairs.items():
            overlapping_rows = [
                row
                for row in predictions
                if row.get("document") == pair[0]
                and type(row.get("source_start")) is int
                and type(row.get("source_end")) is int
                and any(
                    _overlap(
                        int(row["source_start"]),
                        int(row["source_end"]),
                        mention.source_start,
                        mention.source_end,
                    )
                    > 0
                    for mention in mentions
                )
            ]
            if overlapping_rows:
                behavior_pairs.add(pair)
            if any(
                (
                    _technique_family(str(candidate.get("technique_id", "")))
                    if family_level
                    else str(candidate.get("technique_id", ""))
                )
                == pair[1]
                for row in overlapping_rows
                for candidate in row.get("candidates", [])
            ):
                finalist_pairs.add(pair)

        target_count = len(gold_pairs)
        prediction_count = len(predicted_pairs)
        implicit_count = len(implicit_only_pairs)
        return {
            "gold_pairs": target_count,
            "predicted_pairs": prediction_count,
            "correct_pairs": len(correct_pairs),
            "precision": len(correct_pairs) / prediction_count if prediction_count else 0.0,
            "recall": len(correct_pairs) / target_count if target_count else 0.0,
            "implicit_only_gold_pairs": implicit_count,
            "implicit_only_correct_pairs": len(correct_pairs.intersection(implicit_only_pairs)),
            "implicit_only_recall": (
                len(correct_pairs.intersection(implicit_only_pairs)) / implicit_count if implicit_count else 0.0
            ),
            "funnel": {
                "behavior_found": len(behavior_pairs),
                "behavior_recall": len(behavior_pairs) / target_count if target_count else 0.0,
                "finalist_reached": len(finalist_pairs),
                "finalist_recall": len(finalist_pairs) / target_count if target_count else 0.0,
                "conditional_finalist_recall": (len(finalist_pairs) / len(behavior_pairs) if behavior_pairs else 0.0),
                "final_supported": len(correct_pairs),
                "final_recall": len(correct_pairs) / target_count if target_count else 0.0,
            },
        }

    candidate_source_names = (
        [
            "dense_and_late_interaction_and_exemplar",
            "dense_and_late_interaction",
            "dense_and_exemplar",
            "late_interaction_and_exemplar",
            "dense_only",
            "late_interaction_only",
            "exemplar_only",
            "candidate_ledger_unavailable",
        ]
        if exemplar_pool_count
        else [
            "dense_and_late_interaction",
            "dense_only",
            "late_interaction_only",
            "candidate_ledger_unavailable",
        ]
    )
    retrieval_contribution = {
        path: sum(row["retrieval_path"] == path for row in matches) for path in candidate_source_names
    }
    family_retrieval_contribution = {
        path: sum(row["retrieval_path"] == path for row in family_matches) for path in candidate_source_names
    }

    def finalist_ledger(*, family_level: bool) -> list[dict[str, Any]]:
        ledger: list[dict[str, Any]] = []
        for gold in gold_mentions:
            options: list[tuple[dict[str, Any], dict[str, Any]]] = []
            for row in predictions:
                if row.get("document") != gold.document:
                    continue
                if type(row.get("source_start")) is not int or type(row.get("source_end")) is not int:
                    continue
                if (
                    _overlap(
                        int(row["source_start"]),
                        int(row["source_end"]),
                        gold.source_start,
                        gold.source_end,
                    )
                    <= 0
                ):
                    continue
                for candidate in row.get("candidates", []):
                    candidate_id = str(candidate.get("technique_id", ""))
                    matches_gold = (
                        _technique_family(candidate_id) == _technique_family(gold.technique_id)
                        if family_level
                        else candidate_id == gold.technique_id
                    )
                    if matches_gold:
                        options.append((row, candidate))
            if not options:
                continue
            row, candidate = min(
                options,
                key=lambda item: (
                    item[1].get("rerank_rank") if type(item[1].get("rerank_rank")) is int else 10_000,
                    -_overlap(
                        int(item[0]["source_start"]),
                        int(item[0]["source_end"]),
                        gold.source_start,
                        gold.source_end,
                    ),
                ),
            )
            dense_rank = candidate.get("dense_rank")
            late_rank = candidate.get("late_interaction_rank")
            exemplar_rank = candidate.get("exemplar_rank")
            retrieval_path = candidate_source(candidate)
            ledger.append(
                {
                    "mention_id": gold.mention_id,
                    "document": gold.document,
                    "gold_technique_id": gold.technique_id,
                    "candidate_technique_id": candidate.get("technique_id"),
                    "annotation_class": gold.annotation_class,
                    "gold_quote": gold.quote,
                    "predicted_quote": row["quote"],
                    "retrieval_path": retrieval_path,
                    "dense_rank": dense_rank,
                    "late_interaction_rank": late_rank,
                    "exemplar_rank": exemplar_rank,
                    "rerank_rank": candidate.get("rerank_rank"),
                }
            )
        return ledger

    exact_finalists = finalist_ledger(family_level=False)
    family_finalists = finalist_ledger(family_level=True)
    exact_finalist_retrieval_contribution = {
        path: sum(row["retrieval_path"] == path for row in exact_finalists) for path in candidate_source_names
    }
    family_finalist_retrieval_contribution = {
        path: sum(row["retrieval_path"] == path for row in family_finalists) for path in candidate_source_names
    }
    exact_report_technique = report_technique_metrics(family_level=False)
    family_report_technique = report_technique_metrics(family_level=True)
    gates = {
        "suggestion_annotated_span_precision": (annotated_span_precision >= suggestion_annotated_span_precision_gate),
        "report_pair_behavior_recall": (
            exact_report_technique["funnel"]["behavior_recall"] >= report_pair_behavior_recall_gate
        ),
        "report_pair_family_conditional_finalist_recall": (
            family_report_technique["funnel"]["conditional_finalist_recall"]
            >= report_pair_family_conditional_finalist_recall_gate
        ),
        "exact_source_offsets": not invalid_evidence,
        "excluded_documents_absent": not excluded_present,
    }
    return {
        "documents": len(reports),
        "gold": {
            "total": len(gold_mentions),
            "implicit": len(implicit),
            "explicit": len(explicit),
        },
        "predictions": {
            "extracted_behaviors": len(predictions),
            "verifier_supported": sum(
                row.get("support") == "supported" and bool(row.get("selected_technique_id")) for row in predictions
            ),
            "supported_unique": len(supported),
            "supported_on_annotated_spans": len(supported_on_annotated_spans),
            "matched": len(matches),
            "false_positive": len(false_positives),
            "family_matched": len(family_matches),
            "family_false_positive": len(family_false_positives),
            "invalid_evidence": len(invalid_evidence),
        },
        "metrics": {
            "overall_precision": precision,
            "family_precision": family_precision,
            "annotated_span_precision": annotated_span_precision,
            "annotated_span_family_precision": annotated_span_family_precision,
            "overall_recall": len(matches) / len(gold_mentions) if gold_mentions else 0.0,
            "overall_family_recall": len(family_matches) / len(gold_mentions) if gold_mentions else 0.0,
            "implicit_recall": implicit_recall,
            "implicit_family_recall": implicit_family_recall,
            "explicit_recall": matched_explicit / len(explicit) if explicit else 0.0,
            "explicit_family_recall": family_matched_explicit / len(explicit) if explicit else 0.0,
        },
        "funnel": {
            "all": funnel(gold_mentions),
            "implicit": funnel(implicit),
            "explicit": funnel(explicit),
        },
        "report_technique": {
            "exact": exact_report_technique,
            "family": family_report_technique,
        },
        "retrieval_contribution": {
            "dense_pool_count": dense_pool_count,
            "late_interaction_pool_count": late_interaction_pool_count,
            "exemplar_pool_count": exemplar_pool_count,
            "matched_mappings_by_candidate_source": retrieval_contribution,
            "family_matched_mappings_by_candidate_source": family_retrieval_contribution,
            "exact_finalists_by_candidate_source": exact_finalist_retrieval_contribution,
            "family_finalists_by_candidate_source": family_finalist_retrieval_contribution,
        },
        "gates": {
            "thresholds": {
                "suggestion_annotated_span_precision": suggestion_annotated_span_precision_gate,
                "report_pair_behavior_recall": report_pair_behavior_recall_gate,
                "report_pair_family_conditional_finalist_recall": (report_pair_family_conditional_finalist_recall_gate),
            },
            "checks": gates,
            "passed": all(gates.values()),
        },
        "excluded_documents_present": excluded_present,
        "invalid_evidence": invalid_evidence,
        "matches": matches,
        "family_matches": family_matches,
        "exact_finalists": exact_finalists,
        "family_finalists": family_finalists,
        "false_positives": false_positives,
        "family_false_positives": family_false_positives,
        "misses": misses,
    }


def read_predictions(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            rows.append(json.loads(line))
    return rows
