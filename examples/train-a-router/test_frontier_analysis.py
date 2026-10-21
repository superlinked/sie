"""Small exact-sweep and artifact-boundary checks; no models or real outcomes."""

from copy import deepcopy
from fractions import Fraction
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

import frontier_analysis as frontier
import paired_calibration
import paired_training as core
import reproduction
import utility_router as utility


def brute_metrics(policy, p, expected, labels, costs, fees):
    routes = utility.replay(policy, p, expected)
    y, c = np.asarray(labels), np.asarray(costs)
    actual_y = y[np.arange(len(y)), routes.astype(int)]
    actual_c = c[np.arange(len(c)), routes.astype(int)]
    fee = (
        0.0
        if policy["mode"].startswith("always_")
        else None
        if fees is None
        else float(np.mean(fees))
    )
    total = None if fee is None else float(actual_c.mean()) + fee
    return {
        "answer_accuracy_percent": 100 * float(actual_y.mean()),
        "quality_gap_pp": 100 * float((y[:, 1] - actual_y).mean()),
        "answer_usd_per_10000": 10000 * float(actual_c.mean()),
        "router_usd_per_10000": None if fee is None else 10000 * fee,
        "total_usd_per_10000": None if total is None else 10000 * total,
        "strong_share": float(routes.mean()),
        "answer_savings_percent": None
        if not c[:, 1].mean()
        else 100 * (1 - float(actual_c.mean() / c[:, 1].mean())),
        "net_savings_percent": None
        if total is None or not c[:, 1].mean()
        else 100 * (1 - total / float(c[:, 1].mean())),
    }


class SweepTests(unittest.TestCase):
    def assert_metrics(self, actual, expected):
        self.assertEqual(actual.keys(), expected.keys())
        for key, value in expected.items():
            if value is None:
                self.assertIsNone(actual[key], key)
            else:
                self.assertAlmostEqual(actual[key], value, places=9, msg=key)

    def test_exact_sweep_matches_brute_with_opposite_zero_and_negative_costs(self):
        p = [[0, 1], [1, 0], [0.5, 0.5], [0.5, 0.5], [0.2, 0.9], [0.9, 0.2], [0.5, 0.5]]
        expected = [[0, 1], [1, 0], [1, 2], [2, 1], [1, 1], [1, 1], [1, 1]]
        labels = [[0, 1], [1, 0], [1, 0], [0, 1], [1, 0], [0, 1], [1, 1]]
        costs = [[2, 1], [1, 4], [3, 1], [1, 3], [0, 1], [1, 0], [1, 1]]
        candidates = utility.calibrate(p, expected, labels, costs)["candidates"]
        policies = [row["policy"] for row in reversed(candidates)] + [
            candidates[2]["policy"]
        ]
        for fees in (None, [0.01, 0.2, 0, 1, 2, 0.1, 0.05]):
            actual = frontier.sweep_metrics(policies, p, expected, labels, costs, fees)
            for policy, metric in zip(policies, actual, strict=True):
                self.assert_metrics(
                    metric, brute_metrics(policy, p, expected, labels, costs, fees)
                )
        boundary = utility._policy("utility", Fraction(1))
        routes = utility.replay(boundary, p, expected)
        self.assertEqual(routes[:2].tolist(), [True, True])

    def test_adjacent_float_crossings_keep_exact_interval_and_unseen_final_changes(
        self,
    ):
        p = [[0, 0.5], [0.5, 0]]
        costs = [[np.nextafter(1.0, 0), 2], [2, 1]]
        labels = [[0, 1], [1, 0]]
        candidates = utility.calibrate(p, costs, labels, costs)["candidates"]
        policies = [row["policy"] for row in candidates]
        actual = frontier.sweep_metrics(policies, p, costs, labels, costs)
        for policy, metric in zip(policies, actual, strict=True):
            self.assert_metrics(
                metric, brute_metrics(policy, p, costs, labels, costs, None)
            )
        self.assertTrue(
            any(
                tuple(utility.replay(rule, p, costs)) == (False, False)
                for rule in policies
            )
        )
        final_p = [[0.25, 0.5], [0.75, 0.5]]
        actual = frontier.sweep_metrics(policies, final_p, costs, labels, costs)
        for policy, metric in zip(policies, actual, strict=True):
            self.assert_metrics(
                metric, brute_metrics(policy, final_p, costs, labels, costs, None)
            )

    def test_invalid_inputs_and_noncanonical_saved_lambda_fail(self):
        policy = utility._policy("utility", Fraction(1))
        for p, expected, labels, costs, fees in (
            ([[0, float("nan")]], [[1, 2]], [[0, 1]], [[1, 2]], None),
            ([[0, 1]], [[1, 2]], [[0, 2]], [[1, 2]], None),
            ([[0, 1]], [[1, 2]], [[0, 1]], [[1, -1]], None),
            ([[0, 1]], [[1, 2]], [[0, 1]], [[1, 2]], [float("nan")]),
        ):
            with self.assertRaises(ValueError):
                frontier.sweep_metrics([policy], p, expected, labels, costs, fees)
        policy["lambda"] = {"numerator": "2", "denominator": "2"}
        with self.assertRaisesRegex(ValueError, "canonical"):
            frontier.sweep_metrics([policy], [[0, 1]], [[1, 2]], [[0, 1]], [[1, 2]])


class ArtifactTests(unittest.TestCase):
    assert_metrics = SweepTests.assert_metrics

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.data = Path(self.temporary.name) / "data"
        self.run = self.data / "reference"
        self.rows = [
            {
                "id": f"{role}-{i}",
                "role": role,
                "source": "mmlupro",
                "task": "biology",
                "family": f"{role}-family-{i}",
            }
            for role in ("calibration", "final")
            for i in range(3)
        ]
        pair = {
            "pair": "fixture",
            "arms": {"cheap": {"model": "small"}, "strong": {"model": "large"}},
            "cost_scope": "fixture USD",
        }
        self.plan = {
            "source_plan_sha256": "1" * 64,
            "pair_spec_sha256": "2" * 64,
            "acquisition_protocol_sha256": "3" * 64,
            "pair": pair,
            "populations": {
                role: [row["id"] for row in self.rows if row["role"] == role]
                for role in ("calibration", "final")
            },
        }
        self.write(self.run / "plan.json", self.plan)
        self.plan_hash = core.sha(self.run / "plan.json")
        self.p = [[0, 1], [1, 0], [0.5, 0.5]]
        self.expected = [[1, 2], [2, 1], [1, 2]]
        self.labels = [[0, 1], [1, 0], [1, 1]]
        self.costs = [[2, 1], [1, 3], [1, 2]]
        roles = {
            role: {"ids": ids, "probabilities": self.p, "expected_usd": self.expected}
            for role, ids in self.plan["populations"].items()
        }
        topic_roles = deepcopy(roles)
        for value in topic_roles.values():
            value["source_topic_diagnostic"] = deepcopy(value)
        scores = {
            "embedding": {"roles": roles},
            "topic": {"roles": topic_roles},
            "native": {
                "variants": {"base": {"roles": roles}, "lora": {"roles": roles}}
            },
        }
        for method, value in scores.items():
            self.write(
                self.run / f"{method}-scores.json",
                {**value, "plan_sha256": self.plan_hash, "outcomes_accessed": False},
            )
        methods = {
            name: deepcopy(roles)
            for name in (*frontier.LEARNED, "topic_metadata_diagnostic")
        }
        methods["topic_metadata_diagnostic"]["metadata_assisted"] = True
        methods["embedding"]["router_usd"] = {role: [0.1] * 3 for role in roles}
        calibration, bundle = paired_calibration.freeze_arrays(
            methods,
            self.labels,
            self.costs,
            self.plan["populations"]["final"],
            final_rows=[r for r in self.rows if r["role"] == "final"],
        )
        projection = {
            "format": "routerbench-singleton-embedding-cost-projection-v1",
            "model": core.MODEL,
            "hf_revision": core.REVISION,
            "instruction": core.INSTRUCTION,
            "batch_validation": {
                "rows": len(self.rows),
                "every_observed_batch_reproduced": True,
            },
            "singleton_validation": {
                "rows": 90,
                "batches": 90,
                "every_observed_batch_reproduced": True,
            },
            "rows": [
                {"id": r["id"], "role": r["role"], "projected_singleton_usd": "0.1"}
                for r in self.rows
            ],
        }
        self.write(self.data / "fees/projection.json", projection)
        provenance = {
            "plan_sha256": self.plan_hash,
            "source_plan_sha256": self.plan["source_plan_sha256"],
            "pair_spec_sha256": self.plan["pair_spec_sha256"],
            "acquisition_protocol_sha256": self.plan["acquisition_protocol_sha256"],
            "grader_sha256": "4" * 64,
            "calibration_grades_sha256": "5" * 64,
            "training_grades_sha256": "6" * 64,
            "fee_projection_sha256": core.sha(self.data / "fees/projection.json"),
            "scores_sha256": {
                m: core.sha(self.run / f"{m}-scores.json") for m in scores
            },
            "cost_scope": pair["cost_scope"],
            "calibration_source_sha256": core.sha(paired_calibration.__file__),
            "utility_source_sha256": core.sha(utility.__file__),
        }
        self.calibration = {
            "format": "routerbench-paired-calibration-v1",
            "protocol": paired_calibration.PROTOCOL,
            "ids": self.plan["populations"]["calibration"],
            **provenance,
            **calibration,
        }
        self.write(self.run / "frozen/calibration.json", self.calibration)
        self.bundle = {
            **bundle,
            **provenance,
            "calibration_sha256": core.sha(self.run / "frozen/calibration.json"),
        }
        self.write(self.run / "frozen/frozen-final-routes.json", self.bundle)
        final_rows = []
        for row, y, costs in zip(
            [r for r in self.rows if r["role"] == "final"],
            self.labels,
            self.costs,
            strict=True,
        ):
            final_rows.append(
                {
                    "id": row["id"],
                    "role": "final",
                    "dataset": row["source"],
                    "task": row["task"],
                    "family": row["family"],
                    "cheap_correct": y[0],
                    "strong_correct": y[1],
                    "cheap_cost_usd": costs[0],
                    "strong_cost_usd": costs[1],
                }
            )
        self.grades = {
            "format": "routerbench-paired-grades-v1",
            "split": "final",
            "pair": pair["pair"],
            "arms": pair["arms"],
            "source_plan_sha256": provenance["source_plan_sha256"],
            "pair_spec_sha256": provenance["pair_spec_sha256"],
            "cost_scope": provenance["cost_scope"],
            "parent_acquisition_protocol_sha256": provenance[
                "acquisition_protocol_sha256"
            ],
            "parent_grader_sha256": provenance["grader_sha256"],
            "acquisition_protocol_sha256": "7" * 64,
            "grader_sha256": "8" * 64,
            "frozen_final_routing_sha256": core.sha(
                self.run / "frozen/frozen-final-routes.json"
            ),
            "rows": final_rows,
        }
        self.final_path = self.data / "recordings/grades-final.json"
        self.write(self.final_path, self.grades)
        self.checked = patch.object(
            core, "checked_plan", return_value=(self.plan, self.rows)
        )
        self.checked.start()

    def tearDown(self):
        self.checked.stop()
        self.temporary.cleanup()

    def write(self, path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value, sort_keys=True))

    def test_default_build_never_reads_final_grades_or_scores(self):
        self.final_path.unlink()
        for method in ("embedding", "native", "topic"):
            (self.run / f"{method}-scores.json").unlink()
        result = frontier.build(self.data)
        self.assertNotIn("test", result)
        self.assertFalse(result["interpretation"]["test_outcomes_accessed"])
        self.assertFalse(result["interpretation"]["new_blind_evaluation"])
        self.assertNotIn(
            "topic_metadata_diagnostic",
            {row["method"] for row in result["calibration"]},
        )
        expected = (
            sum(
                len(self.calibration["methods"][m]["candidates"])
                for m in frontier.LEARNED
            )
            + 8
        )
        self.assertEqual(len(result["calibration"]), expected)

    def test_loader_checks_plan_calibration_hash_and_cross_binding(self):
        original = self.run / "frozen/calibration.json"
        self.write(original, {**self.calibration, "cost_scope": "different"})
        with self.assertRaisesRegex(ValueError, "calibration file changed"):
            frontier.load_calibration(self.run)
        self.bundle["calibration_sha256"] = core.sha(original)
        self.write(self.run / "frozen/frozen-final-routes.json", self.bundle)
        with self.assertRaisesRegex(ValueError, "provenance differs"):
            frontier.load_calibration(self.run)
        self.write(original, self.calibration)
        self.bundle["calibration_sha256"] = core.sha(original)
        self.bundle["plan_sha256"] = "9" * 64
        self.write(self.run / "frozen/frozen-final-routes.json", self.bundle)
        with self.assertRaisesRegex(ValueError, "run plan changed"):
            frontier.load_calibration(self.run)

    def test_reproduction_loader_uses_inner_calibration_and_retains_envelope(self):
        reproduced = self.data.parent / "run"
        shutil.copytree(self.run, reproduced)
        inner = reproduced / "frozen/internal"
        inner.mkdir()
        for name in ("calibration.json", "frozen-final-routes.json"):
            shutil.move(reproduced / "frozen" / name, inner / name)
        envelope = {
            "format": reproduction.FORMAT,
            "evaluation_mode": reproduction.MODE,
            "frozen_before_final_model_outcomes": False,
            "final_outcomes_accessed_during_calibration": False,
            "generated_routes_path": reproduction.INTERNAL_ROUTES,
            "generated_routes_sha256": core.sha(inner / "frozen-final-routes.json"),
            "original_frozen_routing_sha256": core.sha(
                self.run / "frozen/frozen-final-routes.json"
            ),
            "plan_sha256": self.plan_hash,
        }
        self.write(reproduced / "frozen/frozen-final-routes.json", envelope)
        original_bytes = (reproduced / "frozen/frozen-final-routes.json").read_bytes()
        bundle, calibration = frontier.load_calibration(reproduced)
        self.assertEqual(bundle, self.bundle)
        self.assertEqual(calibration, self.calibration)
        self.assertEqual(
            frontier.build(self.data, reproduced, True)["provenance"][
                "verified_saved_target_masks"
            ],
            26,
        )
        self.assertEqual(
            (reproduced / "frozen/frozen-final-routes.json").read_bytes(),
            original_bytes,
        )

    def test_full_sweep_matches_all_saved_main_policies_and_keeps_original_26(self):
        before = {p: p.read_bytes() for p in self.data.rglob("*.json")}
        result = frontier.build(self.data, include_test=True)
        self.assertEqual(result["provenance"]["verified_saved_target_masks"], 26)
        lookup = {row["point_id"]: row for row in result["test"]}
        for name, policy in self.bundle["policies"].items():
            method = policy["method"]
            if method == "topic_metadata_diagnostic":
                continue
            if method in frontier.LEARNED:
                key = f"{method}:{policy['calibration_candidate_id']}"
                expected = brute_metrics(
                    policy["rule"],
                    self.p,
                    self.expected,
                    self.labels,
                    self.costs,
                    [0.1] * 3 if method == "embedding" else None,
                )
                self.assert_metrics({k: lookup[key][k] for k in expected}, expected)
            elif method == "random":
                point = lookup["random:target-" + str(int(policy["target_gap_pp"]))]
                p = policy["strong_probabilities"][0]
                self.assertAlmostEqual(
                    point["answer_accuracy_percent"],
                    100
                    * np.mean(
                        (1 - p) * np.array(self.labels)[:, 0]
                        + p * np.array(self.labels)[:, 1]
                    ),
                )
        for path, body in before.items():
            self.assertEqual(path.read_bytes(), body)

    def test_final_lineage_score_hash_and_fee_hash_fail_closed(self):
        self.grades["parent_grader_sha256"] = "9" * 64
        self.write(self.final_path, self.grades)
        with self.assertRaisesRegex(ValueError, "parent lineage"):
            frontier.build(self.data, include_test=True)
        self.grades["parent_grader_sha256"] = self.bundle["grader_sha256"]
        self.write(self.final_path, self.grades)
        score_path = self.run / "embedding-scores.json"
        score = core.read(score_path)
        self.write(score_path, {**score, "outcomes_accessed": True})
        with self.assertRaisesRegex(ValueError, "scores changed"):
            frontier.build(self.data, include_test=True)
        self.write(score_path, score)
        self.write(self.data / "fees/projection.json", {})
        with self.assertRaisesRegex(ValueError, "fee projection changed"):
            frontier.build(self.data, include_test=True)

    def test_original_route_mask_tampering_is_detected(self):
        policy = self.bundle["policies"]["lora@0pp"]
        policy["strong_probabilities"][0] = 1 - policy["strong_probabilities"][0]
        self.write(self.run / "frozen/frozen-final-routes.json", self.bundle)
        self.grades["frozen_final_routing_sha256"] = core.sha(
            self.run / "frozen/frozen-final-routes.json"
        )
        self.write(self.final_path, self.grades)
        with self.assertRaisesRegex(ValueError, "decisions differ"):
            frontier.build(self.data, include_test=True)


if __name__ == "__main__":
    unittest.main()
