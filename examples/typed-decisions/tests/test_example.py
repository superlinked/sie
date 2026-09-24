"""Offline tests for the question rendering, the normalisers, the metrics and the request check.

    python3 -m unittest discover -s tests -v

Standard library only, no server, no model and no fetched data: each test builds
the small fixture it needs. Most assert that a specific malformed or forged
recording fails closed rather than turning into a number.
"""

from __future__ import annotations

import contextlib
import copy
import importlib.util
import io
import json
import math
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _module(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


questions = _module("questions")
cvss = _module("cvss")
lanes = _module("lanes")
tuning = _module("tuning")
run = _module("run")
score = _module("score")
tune = _module("tune")
page = _module("page")

SOURCE_QUESTIONS = {
    "action": {
        "type": "choice",
        "instructions": "What should the assistant do next?",
        "criteria": {"answer_directly": "Resolve it now.", "escalate_to_human": "Hand off to a person."},
    },
    "needs_human": {
        "type": "noul",
        "instructions": "This conversation requires a human agent.",
        "criteria": {"false": "Automation can finish it.", "true": "A person must take over."},
    },
    "duplicate": {"type": "noul", "instructions": "This invoice duplicates an earlier one."},
    "urgency": {
        "type": "score",
        "instructions": "How time-sensitive is this?",
        "criteria": ["No time pressure; can wait.", "Routine; normal queue.", "Critical; same day."],
    },
}


def vulnerability_inputs() -> dict[str, Any]:
    return {
        "set": questions.VULNERABILITY_TRIAGE,
        "questions": questions.VULNERABILITY_QUESTIONS,
        "cases": [
            {
                "slug": "CVE-0000-0001",
                "split": "test",
                "state": "Cross-site request forgery in the admin page lets a remote attacker change a password.",
                "gold": {
                    "weakness": "request_forgery",
                    "attack_vector": "network",
                    "user_interaction": "true",
                    "privileges_required": "0",
                    "severity": "2",
                    "needs_account": "false",
                    "remote_unauthenticated": "true",
                    "impact": "limited_change",
                },
            },
            {
                "slug": "CVE-0000-0002",
                "split": "dev",
                "state": "A heap buffer overflow in the image parser allows a local user to crash the program.",
                "gold": {
                    "weakness": "memory_corruption",
                    "attack_vector": "local",
                    "user_interaction": "false",
                    "privileges_required": "1",
                    "severity": "1",
                    "needs_account": "true",
                    "remote_unauthenticated": "false",
                    "impact": "denial_of_service",
                },
            },
            {
                "slug": "CVE-0000-0003",
                "split": "dev",
                "state": "SQL injection in the login form lets a remote attacker read the user table.",
                "gold": {
                    "weakness": "sql_injection",
                    "attack_vector": "network",
                    "user_interaction": "false",
                    "privileges_required": "0",
                    "severity": "3",
                    "needs_account": "false",
                    "remote_unauthenticated": "true",
                    "impact": "full_compromise",
                },
            },
        ],
    }


def typed_call(case: dict[str, Any], variant: str, answers: dict[str, Any]) -> dict[str, Any]:
    phrasing = tuning.phrasing(questions.VULNERABILITY_TRIAGE, "laya", variant)
    requests = run.requests_for("laya", phrasing, case["state"], questions.VULNERABILITY_QUESTIONS)
    return {
        "id": run.call_id(questions.VULNERABILITY_TRIAGE, "laya", variant, case["slug"]),
        "set": questions.VULNERABILITY_TRIAGE,
        "backend": "laya",
        "variant": variant,
        "case": case["slug"],
        "split": case["split"],
        "model": run.model_of("laya"),
        "transport": "sie-extract-typed",
        "status": "ok",
        "requests": requests,
        "responses": [{"data": answers}],
        "timing": {"latency_ms": 10.0},
    }


def perfect_typed_answers(case: dict[str, Any], variant: str) -> dict[str, Any]:
    """A typed response that puts 0.9 on each gold option, in the labels the model was sent."""
    phrasing = tuning.phrasing(questions.VULNERABILITY_TRIAGE, "laya", variant)
    _, label_maps = questions.typed_questions(questions.VULNERABILITY_QUESTIONS, phrasing)
    answers: dict[str, Any] = {}
    for qid, question in questions.VULNERABILITY_QUESTIONS.items():
        gold = case["gold"][qid]
        if question["type"] == questions.NOUL:
            answers[qid] = {"noul": 0.9 if gold == "true" else 0.1}
            continue
        labels = list(label_maps[qid])
        rest = 0.1 / (len(labels) - 1)
        answers[qid] = {"probabilities": {label: 0.9 if label_maps[qid][label] == gold else rest for label in labels}}
    return answers


class QuestionRenderingTests(unittest.TestCase):
    def test_workflow_questions_round_trip_to_the_source(self) -> None:
        converted = {qid: questions.from_source_question(q) for qid, q in SOURCE_QUESTIONS.items()}
        rendered, _ = questions.typed_questions(converted, questions.DESCRIBED)
        self.assertEqual(rendered, SOURCE_QUESTIONS)

    def test_label_options_are_unique_and_true_first(self) -> None:
        for question in [
            *questions.VULNERABILITY_QUESTIONS.values(),
            *map(questions.from_source_question, SOURCE_QUESTIONS.values()),
        ]:
            for variant in questions.VARIANTS:
                pairs = questions.label_options(question, variant)
                labels = [label for _, label in pairs]
                self.assertEqual(len(labels), len(set(labels)))
                self.assertEqual(sorted(key for key, _ in pairs), sorted(questions.keys(question)))
                if question["type"] == questions.NOUL:
                    self.assertEqual(pairs[0][0], "true")

    def test_noul_without_description_uses_statement_and_negation(self) -> None:
        pairs = dict(questions.label_options(questions.from_source_question(SOURCE_QUESTIONS["duplicate"]), "short"))
        self.assertEqual(pairs["true"], "This invoice duplicates an earlier one.")
        self.assertEqual(pairs["false"], "Not the case: this invoice duplicates an earlier one.")

    def test_short_typed_choice_maps_names_back_to_keys(self) -> None:
        rendered, label_maps = questions.typed_questions(questions.VULNERABILITY_QUESTIONS, questions.SHORT)
        self.assertIn("cross-site request forgery", rendered["weakness"]["criteria"])
        self.assertEqual(label_maps["weakness"]["cross-site request forgery"], "request_forgery")
        self.assertNotIn("criteria", rendered["remote_unauthenticated"])


class NormaliserTests(unittest.TestCase):
    def setUp(self) -> None:
        self.case = vulnerability_inputs()["cases"][0]

    def test_typed_answers_map_to_option_keys(self) -> None:
        call = typed_call(self.case, "short", perfect_typed_answers(self.case, "short"))
        answers = score.from_typed(call, questions.VULNERABILITY_QUESTIONS)
        for qid in questions.VULNERABILITY_QUESTIONS:
            gold = self.case["gold"][qid]
            self.assertEqual(answers[qid]["top"], gold, qid)

    def test_typed_unknown_option_fails_closed(self) -> None:
        answers = perfect_typed_answers(self.case, "short")
        answers["weakness"]["probabilities"]["buffer overflow"] = 0.0
        with self.assertRaisesRegex(SystemExit, "was not sent"):
            score.from_typed(typed_call(self.case, "short", answers), questions.VULNERABILITY_QUESTIONS)

    def test_non_finite_and_boolean_probabilities_fail_closed(self) -> None:
        for bad in (float("nan"), True, 1.5):
            answers = perfect_typed_answers(self.case, "described")
            answers["remote_unauthenticated"]["noul"] = bad
            with self.assertRaises(SystemExit):
                score.from_typed(typed_call(self.case, "described", answers), questions.VULNERABILITY_QUESTIONS)

    def test_distribution_that_does_not_sum_to_one_fails_closed(self) -> None:
        answers = perfect_typed_answers(self.case, "described")
        answers["attack_vector"]["probabilities"] = {key: 0.9 for key in answers["attack_vector"]["probabilities"]}
        with self.assertRaisesRegex(SystemExit, "sum to"):
            score.from_typed(typed_call(self.case, "described", answers), questions.VULNERABILITY_QUESTIONS)

    def _label_call(self, classifications: list[list[dict[str, Any]]]) -> dict[str, Any]:
        requests = run.requests_for("gliner2-base", "short", self.case["state"], questions.VULNERABILITY_QUESTIONS)
        return {
            "id": "x",
            "set": questions.VULNERABILITY_TRIAGE,
            "backend": "gliner2-base",
            "variant": "short",
            "requests": requests,
            "responses": [{"classifications": entries} for entries in classifications],
        }

    def test_single_label_answers_keep_only_what_came_back(self) -> None:
        tops = {
            "weakness": "cross-site request forgery",
            "attack_vector": "remotely over the network",
            "remote_unauthenticated": "exploitable remotely without a login",
        }
        call = self._label_call(
            [[{"label": tops[r["question"]], "score": 0.8}] for r in self._label_call([])["requests"]]
        )
        answers = score.from_labels(call, questions.VULNERABILITY_QUESTIONS)
        self.assertIsNone(answers["weakness"]["distribution"])
        self.assertEqual(answers["weakness"]["top"], "request_forgery")
        # Two options: the single answer fixes the other side.
        self.assertAlmostEqual(answers["remote_unauthenticated"]["distribution"]["false"], 0.2)

    def test_label_that_was_not_sent_fails_closed(self) -> None:
        requests = self._label_call([])["requests"]
        call = self._label_call([[{"label": "buffer overflow", "score": 0.9}] for _ in requests])
        with self.assertRaisesRegex(SystemExit, "was not sent"):
            score.from_labels(call, questions.VULNERABILITY_QUESTIONS)


def gold_scores(case: dict[str, Any], qid: str, labels: list[str], variant: str) -> dict[str, float]:
    """Scores that favour the gold option of one question, keyed by the label strings sent."""
    key_of = {label: key for key, label in questions.label_options(questions.VULNERABILITY_QUESTIONS[qid], variant)}
    return {label: 0.8 if key_of[label] == case["gold"][qid] else 0.1 for label in labels}


class GroupedBackendTests(unittest.TestCase):
    def setUp(self) -> None:
        self.case = vulnerability_inputs()["cases"][0]

    def _request(self, backend: str) -> dict[str, Any]:
        (request,) = run.requests_for(backend, "short", self.case["state"], questions.VULNERABILITY_QUESTIONS)
        return request

    def test_gliformer_names_each_group_by_its_question(self) -> None:
        request = self._request("gliformer-large")
        options = request["body"]["params"]["options"]
        weakness = questions.VULNERABILITY_QUESTIONS["weakness"]["instructions"]
        self.assertIn(weakness, options["label_groups"])
        self.assertEqual(request["groups"][weakness], "weakness")
        self.assertEqual((options["multi_label"], options["threshold"]), (True, 0.0))

    def test_grouped_gliclass_keeps_group_names_short_and_the_questions_in_the_instruction(self) -> None:
        request = self._request("gliclass-instruct-large-grouped")
        params = request["body"]["params"]
        self.assertEqual(request["groups"]["remote unauthenticated"], "remote_unauthenticated")
        self.assertIn(
            questions.VULNERABILITY_QUESTIONS["remote_unauthenticated"]["instructions"], params["instruction"]
        )
        self.assertEqual(params["options"]["overflow_policy"], "truncate_text")

    def _gliformer_call(self) -> dict[str, Any]:
        request = self._request("gliformer-large")
        classifications = [
            {"label": f"{name}.{label}", "score": score}
            for name, labels in request["body"]["params"]["options"]["label_groups"].items()
            for label, score in gold_scores(self.case, request["groups"][name], labels, "short").items()
        ]
        return {
            "id": "g",
            "set": questions.VULNERABILITY_TRIAGE,
            "backend": "gliformer-large",
            "variant": "short",
            "requests": [request],
            "responses": [{"classifications": classifications}],
        }

    def test_gliformer_labels_are_split_by_group_and_renormalised(self) -> None:
        answers = score.from_group_labels(self._gliformer_call(), questions.VULNERABILITY_QUESTIONS)
        for qid in questions.VULNERABILITY_QUESTIONS:
            gold = self.case["gold"][qid]
            self.assertEqual(answers[qid]["top"], gold)
            self.assertAlmostEqual(sum(answers[qid]["distribution"].values()), 1.0)

    def test_gliformer_label_below_threshold_counts_as_zero(self) -> None:
        call = self._gliformer_call()
        entries = call["responses"][0]["classifications"]
        dropped = next(e for e in entries if e["score"] < 0.5)
        entries.remove(dropped)
        answers = score.from_group_labels(call, questions.VULNERABILITY_QUESTIONS)
        self.assertIn(0.0, [p for a in answers.values() for p in a["distribution"].values()])

    def test_gliformer_label_from_no_group_fails_closed(self) -> None:
        call = self._gliformer_call()
        call["responses"][0]["classifications"].append({"label": "other question.yes", "score": 0.5})
        with self.assertRaisesRegex(SystemExit, "no group"):
            score.from_group_labels(call, questions.VULNERABILITY_QUESTIONS)

    def _gliclass_grouped_call(self) -> dict[str, Any]:
        request = self._request("gliclass-instruct-large-grouped")
        data = {}
        for name, labels in request["body"]["params"]["options"]["label_groups"].items():
            scores = gold_scores(self.case, request["groups"][name], labels, "short")
            total = sum(scores.values())
            data[name] = {"type": "choice", "probabilities": {label: v / total for label, v in scores.items()}}
        return {
            "id": "c",
            "set": questions.VULNERABILITY_TRIAGE,
            "backend": "gliclass-instruct-large-grouped",
            "variant": "short",
            "requests": [request],
            "responses": [{"data": data}],
        }

    def test_grouped_gliclass_answers_map_to_option_keys(self) -> None:
        answers = score.from_group_answers(self._gliclass_grouped_call(), questions.VULNERABILITY_QUESTIONS)
        for qid in questions.VULNERABILITY_QUESTIONS:
            gold = self.case["gold"][qid]
            self.assertEqual(answers[qid]["top"], gold)

    def test_grouped_gliclass_missing_group_fails_closed(self) -> None:
        call = self._gliclass_grouped_call()
        call["responses"][0]["data"].pop("attack vector")
        with self.assertRaisesRegex(SystemExit, "no answer"):
            score.from_group_answers(call, questions.VULNERABILITY_QUESTIONS)


class PerQuestionBackendTests(unittest.TestCase):
    def setUp(self) -> None:
        self.case = vulnerability_inputs()["cases"][0]

    def test_gliclass_truncates_the_record_rather_than_failing(self) -> None:
        for backend in ("gliclass", "gliclass-base-v1", "gliclass-instruct-large"):
            for request in run.requests_for(backend, "short", self.case["state"], questions.VULNERABILITY_QUESTIONS):
                self.assertEqual(request["body"]["params"]["options"], {"overflow_policy": "truncate_text"})

    def test_gliclass_instruct_sends_the_question_as_its_instruction(self) -> None:
        requests = run.requests_for(
            "gliclass-instruct-large", "short", self.case["state"], questions.VULNERABILITY_QUESTIONS
        )
        self.assertEqual(len(requests), len(questions.VULNERABILITY_QUESTIONS))
        for request in requests:
            question = questions.VULNERABILITY_QUESTIONS[request["question"]]
            self.assertEqual(request["body"]["params"]["instruction"], question["instructions"])


class LaneTests(unittest.TestCase):
    def test_gliformer_is_asked_only_its_questions_on_the_vulnerability_set(self) -> None:
        asked = questions.asked(questions.VULNERABILITY_TRIAGE, "gliformer-large", questions.VULNERABILITY_QUESTIONS)
        self.assertEqual(list(asked), ["weakness", "attack_vector"])
        self.assertEqual(questions.asked(questions.WORKFLOWS, "gliformer-large", {"a": {}}), {"a": {}})

    def test_a_question_the_first_backend_is_not_asked_comes_from_the_second(self) -> None:
        def ans(top: str, other: str, p: float) -> dict[str, Any]:
            return {"distribution": {top: p, other: 1 - p}, "top": top, "top_p": p, "type": "choice"}

        fast = {
            "c1": {
                "weakness": ans("sql_injection", "cross_site_scripting", 0.4),
                "attack_vector": ans("network", "local", 0.95),
            }
        }
        smart = {
            "c1": {
                "weakness": ans("cross_site_scripting", "sql_injection", 0.9),
                "attack_vector": ans("local", "network", 0.9),
                "remote_unauthenticated": ans("true", "false", 0.8),
            }
        }
        lane = {"backend": "f", "escalate_to": "s", "cutoffs": {"weakness": 0.5, "attack_vector": 0.5}}
        answers, escalated = lanes.lane_answers(lane, {"f": fast, "s": smart})
        self.assertEqual(answers["c1"]["weakness"]["top"], "cross_site_scripting")
        self.assertEqual(answers["c1"]["attack_vector"]["top"], "network")
        self.assertEqual(answers["c1"]["remote_unauthenticated"]["top"], "true")
        self.assertEqual(escalated["c1"], {"weakness": True, "attack_vector": False, "remote_unauthenticated": True})


class NLITests(unittest.TestCase):
    def test_statements_go_in_as_hypotheses_and_names_through_the_topic_template(self) -> None:
        case = vulnerability_inputs()["cases"][0]
        for request in run.requests_for(
            "nli-modernbert-base", "short", case["state"], questions.VULNERABILITY_QUESTIONS
        ):
            qtype = questions.VULNERABILITY_QUESTIONS[request["question"]]["type"]
            expected = run.NLI_STATEMENT_TEMPLATE if qtype == questions.NOUL else run.NLI_TOPIC_TEMPLATE
            self.assertEqual(request["body"]["params"]["options"], {"hypothesis_template": expected})


class LayaTests(unittest.TestCase):
    def test_text_and_structured_states_use_the_sie_item_shapes(self) -> None:
        converted = {qid: questions.from_source_question(q) for qid, q in SOURCE_QUESTIONS.items()}
        (text_request,) = run.requests_for("laya", "described", "a ticket", converted)
        (dict_request,) = run.requests_for("laya", "described", {"body": "a ticket"}, converted)
        self.assertEqual(text_request["body"]["items"], [{"text": "a ticket"}])
        self.assertEqual(dict_request["body"]["items"], [{"metadata": {"state": {"body": "a ticket"}}}])
        # The question dict sent is the source's own, exactly as the package receives it.
        self.assertEqual(text_request["body"]["params"]["output_schema"], SOURCE_QUESTIONS)

    def test_package_and_sie_answers_score_the_same(self) -> None:
        case = vulnerability_inputs()["cases"][0]
        sie_call = typed_call(case, "short", perfect_typed_answers(case, "short"))
        package_call = {
            **sie_call,
            "backend": "laya-package",
            "transport": "laya-python-package",
            "requests": run.requests_for("laya-package", "short", case["state"], questions.VULNERABILITY_QUESTIONS),
            "responses": [{"answers": sie_call["responses"][0]["data"]}],
        }
        self.assertEqual(
            score.from_typed(package_call, questions.VULNERABILITY_QUESTIONS),
            score.from_typed(sie_call, questions.VULNERABILITY_QUESTIONS),
        )

    def test_every_planned_backend_is_served_by_sie(self) -> None:
        for backends in run.PLAN.values():
            self.assertFalse([b for b in backends if run.is_package(b)])


class CVSSTests(unittest.TestCase):
    def test_base_scores_match_published_vectors(self) -> None:
        for vector, expected in (
            ("CVSS:3.1/AV:N/AC:L/PR:N/UI:N/S:U/C:H/I:H/A:H", 9.8),
            ("CVSS:3.1/AV:N/AC:L/PR:N/UI:R/S:C/C:L/I:L/A:N", 6.1),
            ("CVSS:3.1/AV:L/AC:L/PR:N/UI:R/S:U/C:H/I:H/A:H", 7.8),
            ("CVSS:3.1/AV:N/AC:L/PR:L/UI:N/S:U/C:H/I:H/A:H", 8.8),
            ("CVSS:3.1/AV:N/AC:L/PR:N/UI:N/S:U/C:N/I:N/A:H", 7.5),
        ):
            self.assertEqual(cvss.base_score(cvss.parse(vector)), expected, vector)

    def test_impact_patterns_and_derived_gold(self) -> None:
        gold = cvss.derived_gold("CVSS:3.1/AV:N/AC:L/PR:L/UI:R/S:C/C:L/I:L/A:N")
        self.assertEqual(gold, {"needs_account": "true", "remote_unauthenticated": "false", "impact": "browser_script"})
        self.assertEqual(
            cvss.impact_pattern(cvss.parse("CVSS:3.1/AV:N/AC:L/PR:N/UI:N/S:U/C:N/I:H/A:H")), "full_compromise"
        )

    def test_composed_severity_is_a_distribution_over_levels(self) -> None:
        certain = cvss.compose_severity({"command_injection": 1.0}, {"network": 1.0}, {"true": 1.0})
        self.assertEqual(max(certain, key=certain.get), "3")
        browser = cvss.compose_severity({"cross_site_scripting": 1.0}, {"network": 1.0}, {"true": 1.0})
        self.assertEqual(max(browser, key=browser.get), "1")
        mixed = cvss.compose_severity(
            {"cross_site_scripting": 0.5, "sql_injection": 0.5},
            {"network": 0.7, "local": 0.3},
            {"true": 0.4, "false": 0.6},
        )
        self.assertAlmostEqual(sum(mixed.values()), 1.0)


class MetricTests(unittest.TestCase):
    def test_ece_of_a_calibrated_and_an_overconfident_set(self) -> None:
        self.assertAlmostEqual(score.ece([(0.75, True), (0.75, True), (0.75, True), (0.75, False)]), 0.0)
        self.assertAlmostEqual(score.ece([(0.95, True), (0.95, False)]), 0.45)

    def test_macro_f1_and_brier(self) -> None:
        self.assertAlmostEqual(score.macro_f1(["a", "a", "b"], ["a", "b", "b"]), (2 / 3 + 2 / 3) / 2)
        self.assertAlmostEqual(score.brier({"a": 0.5, "b": 0.5}, "a"), 0.5)

    def test_question_metrics_reports_majority_and_levels(self) -> None:
        rows = [
            ({"type": "score", "top": "2", "top_p": 0.6, "distribution": {"0": 0.1, "1": 0.3, "2": 0.6}}, "2"),
            ({"type": "score", "top": "0", "top_p": 0.5, "distribution": {"0": 0.5, "1": 0.25, "2": 0.25}}, "2"),
        ]
        metrics = score.question_metrics(rows)
        self.assertEqual(metrics["accuracy"], 0.5)
        self.assertEqual(metrics["majority"]["accuracy"], 1.0)
        self.assertAlmostEqual(metrics["mae"], (abs(1.5 - 2) + abs(0.75 - 2)) / 2)
        self.assertNotIn("within_one", metrics)
        self.assertTrue(math.isfinite(metrics["brier"]))

    def test_variant_tie_goes_to_described(self) -> None:
        dev = {v: {"overall": {"accuracy": 0.5}} for v in questions.VARIANTS}
        self.assertEqual(score.choose_variant(dev), questions.DESCRIBED)
        dev["short"]["overall"]["accuracy"] = 0.6
        self.assertEqual(score.choose_variant(dev), questions.SHORT)


class TuneTests(unittest.TestCase):
    def test_a_phrasing_that_passes_beats_a_higher_one_that_does_not(self) -> None:
        tried = {
            "described": {"passes": False, "p05_balanced_accuracy": 0.69, "balanced_accuracy": 0.80},
            "short": {"passes": True, "p05_balanced_accuracy": 0.71, "balanced_accuracy": 0.75},
            "concrete": {"passes": True, "p05_balanced_accuracy": 0.74, "balanced_accuracy": 0.74},
        }
        self.assertEqual(tune.best_phrasing(tried), "concrete")
        for entry in tried.values():
            entry["passes"] = False
        self.assertEqual(tune.best_phrasing(tried), "concrete")
        tried["short"]["p05_balanced_accuracy"] = 0.74
        tried["short"]["balanced_accuracy"] = 0.74
        tried["concrete"]["balanced_accuracy"] = 0.74
        self.assertEqual(tune.best_phrasing(tried), "short")

    def test_balanced_accuracy_leaves_out_options_with_few_records(self) -> None:
        gold = ["a"] * 10 + ["b"] * 10 + ["c"]
        predicted = ["a"] * 10 + ["b"] * 5 + ["a"] * 5 + ["a"]
        self.assertAlmostEqual(score.balanced_accuracy(gold, predicted), 0.75)
        self.assertEqual(tune.worst_recall(tune.metrics(gold, predicted)), 0.5)


class CardTests(unittest.TestCase):
    def test_cards_skip_misses_overridden_answers_and_repeated_weakness_classes(self) -> None:
        def ans(top: str, p: float, returned: str | None = None) -> dict[str, Any]:
            return {"top": top, "top_p": p, "returned": {"top": returned or top, "top_p": p}}

        gold = {
            f"CVE-{i}": {"weakness": w, "attack_vector": "network", "remote_unauthenticated": "true"}
            for i, w in enumerate(("sql_injection", "sql_injection", "path_traversal", "file_upload"))
        }
        order = sorted(gold, key=lambda slug: page.sha256(page.CARD_SALT + slug))
        smart = {
            slug: {"weakness": ans(g["weakness"], 0.9), "attack_vector": ans("network", 0.9)}
            for slug, g in gold.items()
        }
        fast = {slug: {"attack_vector": ans("network", 0.8)} for slug in gold}
        # The first record in card order is answered wrong; the second has an answer the rule changed.
        smart[order[0]]["attack_vector"] = ans("local", 0.6)
        fast[order[1]]["attack_vector"] = ans("network", 0.4, returned="local")
        board = {"fast": {"cells": {"attack_vector": {}}}, "smart": {"cells": {"weakness": {}, "attack_vector": {}}}}
        cards = page.choose_cards(gold, {"fast": fast, "smart": smart}, board)
        self.assertEqual(len(cards), 1)
        self.assertEqual(cards[0]["card"], "remote-no-login")
        self.assertNotIn(cards[0]["case"], order[:2])
        self.assertEqual(set(cards[0]["answers"]["smart"]), {"weakness", "attack_vector"})


class CreditTests(unittest.TestCase):
    PRICES = {
        "pricing_version": "test",
        "stripe_credit_pack": {"usd_cents": 5000, "credits": 5000000},
        "items": [
            {
                "model": "Qwen/Qwen3.5-4B",
                "operation": "generate",
                "profile": "default",
                "unit": unit,
                "usd_per_unit": {"numerator": numerator, "denominator": "1000000"},
            }
            for unit, numerator in (("input_tokens", "1"), ("output_tokens", "2"))
        ],
    }

    def test_a_listed_chat_model_is_priced_from_its_recorded_usage(self) -> None:
        calls = [
            {
                "backend": "llm",
                "model": "Qwen/Qwen3.5-4B:no-spec",
                "responses": [{"usage": {"prompt_tokens": 100, "completion_tokens": 50}}],
            }
        ]
        figure = page.credits({"backend": "llm"}, calls, self.PRICES)
        # 100 x $1e-6 + 50 x $2e-6 = $0.0002, at 100,000 credits per dollar.
        self.assertAlmostEqual(figure["median_credits_per_record"], 20.0)
        self.assertAlmostEqual(figure["usd_per_1000_records"], 0.2)

    def test_an_unlisted_model_gets_no_figure(self) -> None:
        calls = [{"backend": "fast", "model": "knowledgator/gliformer-large-v1", "responses": [{}]}]
        self.assertIsNone(page.credits({"backend": "fast"}, calls, self.PRICES))


TUNED_LAYA = {
    qid: {"phrasing": phrasing}
    for qid, phrasing in zip(
        questions.VULNERABILITY_QUESTIONS,
        ("concrete", "short", "described"),
        strict=True,
    )
}


class CheckTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.data = Path(self.tmp.name)
        self.saved_tuning_path = tuning.TUNING_PATH
        tuning.TUNING_PATH = self.data / "tuning.json"
        tuning.TUNING_PATH.write_text(
            json.dumps({"backends": {questions.VULNERABILITY_TRIAGE: {"laya": {"phrasing": TUNED_LAYA}}}}),
            encoding="utf-8",
        )
        inputs = vulnerability_inputs()
        path = self.data / f"inputs/{questions.VULNERABILITY_TRIAGE}/cases.json"
        path.parent.mkdir(parents=True)
        path.write_text(json.dumps(inputs), encoding="utf-8")
        plan = run.PLAN[questions.VULNERABILITY_TRIAGE]["laya"]
        self.calls = [
            typed_call(case, variant, perfect_typed_answers(case, variant))
            for case in inputs["cases"]
            for variant in plan[case["split"]]
        ]

    def tearDown(self) -> None:
        tuning.TUNING_PATH = self.saved_tuning_path
        self.tmp.cleanup()

    def test_tuned_calls_use_the_tuned_phrasing(self) -> None:
        tuned = [call for call in self.calls if call["variant"] == tuning.TUNED]
        self.assertTrue(tuned)
        sent = tuned[0]["requests"][0]["body"]["params"]["output_schema"]
        self.assertIn("buffer overflow or out-of-bounds memory access", sent["weakness"]["criteria"])
        self.assertIn("criteria", sent["remote_unauthenticated"])

    def _check(self, calls: list[dict[str, Any]]) -> int:
        path = self.data / "calls.json"
        path.write_text(json.dumps({"calls": calls}), encoding="utf-8")
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            return run.check(self.data, [path], partial=True)

    def test_recorded_requests_rebuild_from_the_inputs(self) -> None:
        self.assertEqual(self._check(self.calls), 0)

    def test_missing_call_fails(self) -> None:
        self.assertEqual(self._check(self.calls[:-1]), 1)

    def test_duplicate_call_fails(self) -> None:
        self.assertEqual(self._check([*self.calls, self.calls[0]]), 1)

    def test_edited_request_fails(self) -> None:
        calls = copy.deepcopy(self.calls)
        calls[0]["requests"][0]["body"]["items"][0]["text"] += " Edited."
        self.assertEqual(self._check(calls), 1)

    def test_digested_requests_check_and_an_edited_digest_fails(self) -> None:
        calls = [{**call, "requests": run.digested(call["requests"])} for call in self.calls]
        self.assertNotIn("body", calls[0]["requests"][0])
        self.assertEqual(self._check(calls), 0)
        calls[0]["requests"][0]["body_sha256"] = "0" * 64
        self.assertEqual(self._check(calls), 1)

    def test_wrong_model_fails(self) -> None:
        calls = copy.deepcopy(self.calls)
        calls[0]["model"] = "someone/else"
        self.assertEqual(self._check(calls), 1)

    def test_scoring_refuses_failed_calls(self) -> None:
        calls = copy.deepcopy(self.calls)
        calls[0]["status"] = "error"
        with self.assertRaisesRegex(SystemExit, "failed"):
            score.usable({"calls": calls})


if __name__ == "__main__":
    unittest.main()
