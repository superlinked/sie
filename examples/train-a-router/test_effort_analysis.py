"""Small offline checks for effort grading, accounting and exact fixed-route replay."""

import hashlib
import tempfile
import unittest
from copy import deepcopy
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

import collect
from effort import analyze, collect_low
from evaluation import frontier, utility_router


def item(identity="x", request="Return exactly yes."):
    return {"id": identity, "request": request, "response_contract": {"type": "exact_string", "max_length": 2048}}


def response(question, answer="yes", *, truncated=False):
    body = collect_low.body_for(question)
    usage = {"input_tokens": 100, "output_tokens": 25,
             "input_tokens_details": {"cached_tokens": 10, "cache_write_tokens": 20},
             "output_tokens_details": {"reasoning_tokens": 20}, "total_tokens": 125}
    return {"id": question["id"], "arm": "luna", "model": "gpt-6-luna", "reasoning_effort": "low",
            "body_sha256": collect_low.digest(body), "status": "complete", "http_status": 200,
            "created_unix": 1000., "terminal_response": True, "execution_unknown": False,
            "resolved_model": "gpt-6-luna", "service_tier": "default", "resolved_reasoning_effort": "low",
            "provider_status": "incomplete" if truncated else "completed", "elapsed_seconds": 1.25,
            "usage": usage, "cost": collect.openai_cost(usage, body),
            "messages": [{"type": "output_text", "text": analyze.json.dumps({"answer": answer})}],
            "answer": None if truncated else answer, "valid_answer": not truncated}


def collection(root, questions, attempts):
    """Produce explicit immutable physical receipts, never run a transport/scheduler."""
    plan = {"protocol": collect_low.PROTOCOL, "budget_usd": "20.00"}
    analyze.write(root / "plan.json", plan)
    plan_sha, entries = analyze.sha(root / "plan.json"), []
    amounts = [Decimal(0), Decimal(0), Decimal(0)]
    for question, records in zip(questions, attempts, strict=True):
        body = collect_low.body_for(question)
        reserve = collect.reservation(body, "luna")
        state = {"id": question["id"], "status": "uncertain", "selected_attempt": None, "attempts": []}
        for number, saved in enumerate(records, 1):
            prefix = f"attempts/{hashlib.sha256(question['id'].encode()).hexdigest()}/{number:03d}"
            (root / prefix).parent.mkdir(parents=True, exist_ok=True)
            intent = {"id": question["id"], "arm": "luna", "model": "gpt-6-luna", "reasoning_effort": "low",
                      "body_sha256": collect_low.digest(body), "plan_sha256": plan_sha, "attempt": number,
                      "reserved_usd": str(reserve), "created_unix": 1000.}
            analyze.write(root / (prefix + "-intent.json"), intent)
            intent_sha = analyze.sha(root / (prefix + "-intent.json"))
            row = deepcopy(saved) if saved is not None else {
                "status": "interrupted_unknown", "execution_unknown": True, "terminal_response": False}
            row.update(intent_sha256=intent_sha, attempt=number)
            if saved is not None:
                analyze.write(root / (prefix + "-response.json"), row)
            amount, budget, unknown = collect_low.accounting(row, reserve, body)
            amounts = [a + b for a, b in zip(amounts, (amount, budget, unknown), strict=True)]
            action = collect_low.decision(row, number)
            receipt = {"attempt": number, "intent_path": prefix + "-intent.json", "intent_sha256": intent_sha,
                       "record_path": prefix + "-response.json" if saved is not None else None,
                       "record_sha256": analyze.sha(root / (prefix + "-response.json")) if saved is not None else None,
                       "status": row["status"], "action": action,
                       "known_list_usd": None if unknown else str(amount),
                       "known_budget_usd": None if unknown else str(budget), "unknown_reservation_usd": str(unknown)}
            state["attempts"].append(receipt)
            state["status"] = "complete" if action == "terminal" and row["status"] == "complete" else action
            if row.get("terminal_response"):
                state["selected_attempt"] = number
        entries.append(state)
    selection = {"format": "bbeh-luna-low-selection-v1", "plan_sha256": plan_sha, "role": "final", "arm": "luna",
                 "model": "gpt-6-luna", "reasoning_effort": "low", "ordered_ids": [q["id"] for q in questions],
                 "entries": entries, "first_terminal_response_only": True, "no_quality_retries": True}
    analyze.write(root / "selection.json", selection)
    complete = sum(s["status"] == "complete" for s in entries)
    summary = {"format": "bbeh-luna-low-collection-v1", "plan_sha256": plan_sha,
               "selection_sha256": analyze.sha(root / "selection.json"), "expected_responses": len(entries),
               "physical_attempts": sum(len(s["attempts"]) for s in entries),
               "selected_responses": sum(s["selected_attempt"] is not None for s in entries),
               "complete_receipts": complete, "status_counts": dict(analyze.Counter(s["status"] for s in entries)),
               "known_list_usd": str(amounts[0]), "known_budget_usd": str(amounts[1]),
               "unknown_reservation_usd": str(amounts[2]), "cost_upper_bound_usd": str(amounts[1] + amounts[2]),
               "stop": None, "status": "complete" if complete == len(entries) else "incomplete"}
    analyze.write(root / "collection.json", summary)
    return plan


OFFICIAL = SimpleNamespace(preprocess_sample=str.strip, preprocess_reference=str.strip,
                           evaluate_correctness=lambda a, b: a.strip() == b.strip())


def metric(correct, cost="0.01", elapsed=1.):
    return {"correct": correct, "official_raw_correct": correct, "literal_correct": correct,
            "valid": True, "truncated": False, "status": "complete", "provider_status": "completed",
            "available_answer": True, "cost_usd": cost, "cost_upper_bound_usd": cost,
            "elapsed_seconds": elapsed, "usage": {"input_tokens": 10, "output_tokens": 6,
                "input_tokens_details": {"cached_tokens": 0, "cache_write_tokens": 0},
                "output_tokens_details": {"reasoning_tokens": 4}}}


class EffortAnalysisTests(unittest.TestCase):
    def test_valid_truncated_and_missing_are_retained_with_cost_bounds(self):
        questions = [item("a"), item("b"), item("c")]
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            plan = collection(root, questions, [[response(questions[0])], [response(questions[1], truncated=True)], [None]])
            chosen, ledger, summary = analyze.checked_attempts(root, plan, questions)
            graded = [analyze.grade_low(q, {"task": "test"}, "yes", chosen[q["id"]], OFFICIAL) for q in questions]
            self.assertEqual([r["correct"] for r in graded], [1, 0, 0])
            self.assertEqual(len(ledger), 3)
            self.assertEqual(summary["complete_receipts"], 2)  # Includes provider-truncated terminal receipt.
            self.assertTrue(graded[1]["truncated"])
            result = analyze.summarize(graded)
            self.assertEqual(result["n"], 3)
            self.assertIsNone(result["answer_usd_per_10000"])
            self.assertEqual(result["cost_missing_n"], 1)
            self.assertGreater(Decimal(result["selected_cost_bounds_usd"]["upper"]),
                               Decimal(result["known_selected_cost_usd"]))
            self.assertEqual(result["selected_elapsed_seconds"]["missing_n"], 1)

    def test_paid_retry_ledger_is_separate_from_selected_answer(self):
        question = item()
        failed = {**response(question), "http_status": 503, "terminal_response": False,
                  "status": "provider_error", "execution_unknown": False, "cost": None, "usage": None}
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            plan = collection(root, [question], [[failed, response(question)]])
            chosen, ledger, summary = analyze.checked_attempts(root, plan, [question])
            graded = analyze.grade_low(question, {"task": "test"}, "yes", chosen["x"], OFFICIAL)
            self.assertEqual(len(ledger), 2)
            self.assertEqual(graded["cost_usd"], response(question)["cost"]["usd_estimate"])
            self.assertGreater(Decimal(summary["unknown_reservation_usd"]), 0)
            self.assertEqual(summary["status"], "complete")

    def test_hash_tampering_and_terminal_retry_fail_closed(self):
        question = item()
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            plan = collection(root, [question], [[response(question)]])
            path = next((root / "attempts").rglob("*-response.json"))
            path.write_text("{}", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Changed or missing artifact"):
                analyze.checked_attempts(root, plan, [question])
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            plan = collection(root, [question], [[response(question), response(question)]])
            with self.assertRaisesRegex(ValueError, "retried"):
                analyze.checked_attempts(root, plan, [question])

    def test_normalization_and_decimal_do_not_use_binary_float_equivalence(self):
        q = item(request="Pick one.\n(A) red\n(B) blue")
        selected = {"response": response(q, "(B) blue"), "attempt": {"record_sha256": "x"},
                    "state": {"attempts": [1]}, "unknown_bound_usd": "0"}
        graded = analyze.grade_low(q, {"task": "shuffled_objects"}, "(B)", selected, OFFICIAL)
        self.assertEqual(graded["correct"], 1)
        self.assertEqual(graded["literal_correct"], 0)
        self.assertFalse(analyze.grade_v2.decimal_match(OFFICIAL, "9007199254740992", "9007199254740993"))

    def test_stratified_whole_family_bootstrap_preserves_task_weights(self):
        rows = [{"task": "a", "family": "family-a"}] * 3 + [{"task": "b", "family": "family-b"}]
        bootstrap = frontier.Bootstrap(rows, 20)
        values = np.array([[0.], [0.], [0.], [1.]])
        np.testing.assert_array_equal(bootstrap.means(values), np.full((20, 1), .25))
        same = [metric(x) for x in [0, 1, 1, 0]]
        comparison = analyze.compare(same, same, bootstrap)
        self.assertEqual(comparison["accuracy_difference_pp_ci95"]["lower"], 0)
        self.assertEqual(comparison["answer_usd_per_10000_difference_ci95"]["upper"], 0)
        with self.assertRaisesRegex(ValueError, "spans tasks"):
            frontier.Bootstrap([{"task": "a", "family": "x"}, {"task": "b", "family": "x"}], 2)

    def test_exact_fraction_ties_signed_cost_and_bypasses(self):
        p, c = [[.25, .5], [.5, .25], [.5, .5]], [[0, .25], [.25, 0], [0, 0]]
        rule = {"format": utility_router.FORMAT, "mode": "utility", "lambda": {"numerator": "1", "denominator": "1"}}
        self.assertEqual(utility_router.replay(rule, p, c).tolist(), [True, True, True])
        rule["lambda"] = {"numerator": "1000000000000000001", "denominator": "1000000000000000000"}
        self.assertEqual(utility_router.replay(rule, p, c).tolist(), [False, True, True])
        for mode, expected in (("cost_limit", [False, True, True]), ("always_cheap", [False] * 3),
                               ("always_strong", [True] * 3)):
            self.assertEqual(utility_router.replay({"format": utility_router.FORMAT, "mode": mode}, p, c).tolist(), expected)

    def test_token_accounting_does_not_add_reasoning_twice(self):
        q = item()
        r = response(q)
        # 70 fresh input,10 cached,20 written,25 output including20 reasoning.
        expected = (Decimal(70) * Decimal('.10') + Decimal(10) * Decimal('.01')
                    + Decimal(20) * Decimal('.125') + Decimal(25) * Decimal('.50')) / 1_000_000
        self.assertEqual(Decimal(r["cost"]["usd_estimate"]), expected)
        old = {"cheap_" + k: v for k, v in metric(1).items()}
        old.update(cheap_usage=r["usage"], cheap_cost_usd=float(expected))
        self.assertEqual(analyze.original_arm(old, "cheap", q)["cost_usd"], str(expected))

    def test_replay_keeps_masks_and_checks_published_values(self):
        rows = [{"id": str(i), "task": "a", "family": str(i),
                 "luna_medium": metric(i % 2), "luna_low": metric(1, "0.005", .5),
                 "sol_medium": metric(1, "0.05", 2.)} for i in range(4)]
        p = [[.7, .8], [.5, .8], [.3, .8], [.1, .8]]
        c = [[.01, .05]] * 4
        rule = {"format": utility_router.FORMAT, "mode": "utility", "lambda": {"numerator": "10", "denominator": "1"}}
        routes = utility_router.replay(rule, p, c)
        candidate = {"candidate_id": 3, "policy": rule, "calibration": {"fixed": True}}
        methods, choices, report = {}, {}, {}
        for name in analyze.METHODS:
            embedding = name in ("embedding", "embedding__textcost", "knn")
            methods[name] = {"scores": {"probabilities": p, "expected_usd": c, "metadata_assisted": False,
                                         "router_kind": "embedding" if embedding else "native"},
                             "candidates": [candidate], "calibration_frontier_candidate_ids": [3]}
            choices[name] = {"0.8": candidate, "0.6": candidate}
            old = [r["sol_medium" if mask else "luna_medium"] for r, mask in zip(rows, routes, strict=True)]
            metrics = analyze.route_metrics(old, routes, [.001] * 4 if embedding else None)
            report[name] = {"points": [{"candidate_id": 3, "rule": rule,
                                        "answer_accuracy_percent": metrics["accuracy_percent"],
                                        **{k: metrics[k] for k in ("strong_share", "answer_usd_per_10000",
                                                                   "router_usd_per_10000", "total_usd_per_10000")}}]}
        original = {"frozen": {"methods": methods, "encoder_fees_usd": [.001] * 4},
                    "choices": {"choices": choices}, "report": {"methods": report}}
        results = analyze.replay_policies(original, rows, frontier.Bootstrap(rows, 20))
        self.assertEqual(len(results), 9)
        for point in results:
            self.assertEqual(point["original_medium"]["strong_share"], point["substituted_low"]["strong_share"])
            self.assertEqual(point["original_medium"]["router_usd_per_10000"], point["substituted_low"]["router_usd_per_10000"])
        original["report"]["methods"]["base"]["points"][0]["strong_share"] += .01
        with self.assertRaisesRegex(ValueError, "replay parity"):
            analyze.replay_policies(original, rows, frontier.Bootstrap(rows, 20))

    def test_pinned_manifest_is_checked_before_any_grades(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            analyze.write(root / "manifest.json", {"files": []})
            with patch.object(analyze, "read", side_effect=AssertionError("Must reject before grade reads")), \
                    self.assertRaisesRegex(ValueError, "manifest does not match"):
                analyze.checked_data(root)


if __name__ == "__main__":
    unittest.main()
