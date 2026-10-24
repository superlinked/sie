"""Synthetic comparison checks: no models, providers or published answer data."""

import copy
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

import training_comparison as comparison
import workflow
from embedding import recipes as core
from embedding.cost_iteration import refit
from evaluation import frontier
from test_workflow import make_run, replace_json


def candidate(identifier, accuracy, cost, final_accuracy=0):
    return {
        "candidate_id": identifier, "policy": {"mode": "always_cheap"},
        "calibration": {"answer_accuracy_percent": accuracy, "answer_cost_usd_per10000": cost},
        "final": {"answer_accuracy_percent": final_accuracy},
    }


def fixture(root):
    """Construct a small real source/score/calibration chain using synthetic rows."""
    run, plan, rows, grades, _ = make_run(root)
    data = root / "data"
    reference = data / "reference"
    reference.mkdir(parents=True)
    shutil.move(run, reference / "embedding")
    for role, path in grades.items():
        destination = data / f"grades/grades-{role}.json"
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, destination)
    template = core.read(grades["train"])
    for role in ("calibration", "final"):
        values = []
        for i, row in enumerate(rows[role]):
            values.append({**template["rows"][i], "id": row["id"],
                           "cheap_cost_usd": "0.001", "strong_cost_usd": "0.010"})
        core.write(data / f"grades/grades-{role}.json", {**template, "role": role, "rows": values})
    core.write(data / "manifest.json", {"fixture": "synthetic-only"})
    core.write(data / "fees.json", {
        "source_plan_sha256": plan["source_plan_sha256"], "model": core.MODEL,
        "hf_revision": core.REVISION, "instruction": core.INSTRUCTION,
        "cost_basis": "validated_singleton_projection",
        "rows": [{"id": row["id"], "projected_singleton_usd": "0.00001"}
                 for role_rows in rows.values() for row in role_rows],
    })
    native = root / "native-run"
    native_plan = {"populations": {role: [r["id"] for r in value] for role, value in rows.items()}}
    core.write(reference / "native/plan.json", native_plan)
    core.write(native / "plan.json", native_plan)
    source = root / "synthetic-source"
    implementations = {}
    for name in ("fast_protocol.py", "fast_runtime.py", "fast_train.py"):
        path = source / "native" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"# Synthetic {name}\n", encoding="utf-8")
        implementations[name] = core.sha(path)
    recipe = {
        "format": "laya-fast-runtime-recipe-v1", "recipe_id": "laya-fa2-grouped-v1",
        "reference_plan_sha256": core.sha(native / "plan.json"),
        "implementation_sources_sha256": implementations, "final_outcomes_accessed": False,
    }
    core.write(native / "native/fast-recipe.json", recipe)
    core.write(native / "native/fast-runtime.json", {"fixture": "no-model-runtime"})
    trained = {
        "plan_sha256": core.sha(native / "plan.json"),
        "training_ids": native_plan["populations"]["train"] + native_plan["populations"]["development"],
        "selection_train_ids": native_plan["populations"]["train"],
        "selection_development_ids": native_plan["populations"]["development"],
        "training_grades_sha256": core.sha(data / "grades/grades-train.json"),
        "development_grades_sha256": core.sha(data / "grades/grades-development.json"),
        "fast_recipe_sha256": core.sha(native / "native/fast-recipe.json"),
        "fast_runtime_sha256": core.sha(native / "native/fast-runtime.json"),
        "fast_implementation_sources_sha256": implementations,
        "elapsed_seconds": 12, "selected_epoch": 3,
    }
    core.write(native / "native/trained.json", trained)
    core.write(reference / "native/native/trained.json", {"elapsed_seconds": 40, "selected_epoch": 2})
    paths = {}
    for role in ("calibration", "final"):
        embedding = core.read(reference / f"embedding/scores-{role}.json")
        method = {**embedding["methods"]["embedding"], "router_kind": "native"}
        old = {**embedding, "model_sha256": core.sha(reference / "native/native/trained.json"),
               "methods": {"base": method, "lora": copy.deepcopy(method)}}
        core.write(reference / f"native/native/scores-{role}.json", old)
        core.write(native / f"native/scores-{role}.json", {
            **old, "model_sha256": core.sha(native / "native/trained.json"),
        })
        cost = {
            "format": "bbeh-shared-input-cost-scores-v1", "role": role,
            "ids": embedding["ids"], "source_plan_sha256": plan["source_plan_sha256"],
            "pair_sha256": plan["pair_sha256"], "cost_model_sha256": "7" * 64,
            "promotion_sha256": refit.PROMOTION_SHA, "outcomes_accessed": False,
            "expected_usd": [[0.001, 0.010], [0.002, 0.020]],
        }
        core.write(reference / f"cost/costs-{role}.json", cost)
        paths[role] = []
        for prefix, original in (("", reference / f"embedding/scores-{role}.json"),
                                 ("native-", reference / f"native/native/scores-{role}.json")):
            output = reference / f"cost/{prefix}scores-{role}.json"
            refit.augment_files(original, reference / f"cost/costs-{role}.json", output,
                                reference / f"cost/{prefix}combined-model.json")
            paths[role].append(output)
    calibration = frontier.calibrate(reference / "embedding", data / "grades/grades-calibration.json",
                                      paths["calibration"], data / "fees.json",
                                      reference / "calibration-frontiers.json")
    frontier.freeze(reference / "embedding", reference / "calibration-frontiers.json", paths["final"],
                    data / "fees.json", reference / "frozen-frontiers.json")
    choices = {}
    for method in comparison.METHODS.values():
        choices[method] = {}
        for cap in comparison.CAPS:
            selected = comparison.choose(calibration["methods"][method]["frontier"], 100 * cap)
            choices[method][str(cap)] = None if selected is None else {
                key: selected[key] for key in ("candidate_id", "policy", "calibration")
            }
    core.write(data / "metadata/presentation-policies.json", {
        "calibration_sha256": core.sha(reference / "calibration-frontiers.json"),
        "frozen_sha256": core.sha(reference / "frozen-frontiers.json"),
        "illustrative_caps_fraction_of_always_strong_answer_cost": list(comparison.CAPS),
        "choices": choices,
    })
    return data, native, root / "comparison", source


class CalibrationChoiceTests(unittest.TestCase):
    def test_only_calibration_metrics_choose_policy_with_deterministic_ties(self):
        points = [candidate(9, 80, 40, 100), candidate(8, 80, 30, 0),
                  candidate(2, 80, 30, 1), candidate(1, 99, 41, 100)]
        self.assertEqual(comparison.choose(points, 40)["candidate_id"], 2)
        for point in points:
            point["final"]["answer_accuracy_percent"] = 100 - point["final"]["answer_accuracy_percent"]
        self.assertEqual(comparison.choose(points[::-1], 40)["candidate_id"], 2)

    def test_infeasible_budget_returns_no_policy_and_boundary_is_inclusive(self):
        points = [candidate(1, 70, 20)]
        self.assertIsNone(comparison.choose(points, 19.99))
        self.assertIsNone(comparison.choose([], 100))
        self.assertIs(comparison.choose(points, 20), points[0])


class PairedDifferenceTests(unittest.TestCase):
    def test_identical_routes_have_exact_zero_deltas_and_intervals(self):
        rows = [{"task": "a", "family": str(i // 2)} for i in range(8)]
        y = np.array([[0, 1], [1, 0]] * 4)
        cost = np.array([[1, 3], [2, 4]] * 4)
        route = np.array([False, True] * 4)
        result = comparison.paired_differences(rows, y, cost, route, route.copy())
        for metric in result.values():
            self.assertEqual(metric["estimate"], 0)
            self.assertEqual(metric["ci95"], {"lower": 0, "upper": 0, "defined_draws": 2000, "total_draws": 2000})

    def test_delta_units_sign_and_family_resampling(self):
        rows = [{"task": "a", "family": str(i // 10)} for i in range(20)]
        y = np.array([[1, 0]] * 10 + [[0, 1]] * 10)
        cost = np.tile([1, 4], (20, 1))
        result = comparison.paired_differences(rows, y, cost, np.ones(20, bool), np.zeros(20, bool))
        self.assertEqual(result["accuracy_pp"]["estimate"], 0)
        self.assertEqual(result["accuracy_pp"]["ci95"]["lower"], -100)
        self.assertEqual(result["accuracy_pp"]["ci95"]["upper"], 100)
        self.assertEqual(result["answer_usd_per_10000"]["estimate"], -30000)
        self.assertEqual(result["answer_savings_pp"]["estimate"], 75)
        self.assertEqual(result["strong_share_pp"]["estimate"], -100)

    def test_observed_task_weights_are_fixed_and_zero_strong_cost_is_rejected(self):
        rows = [{"task": "a", "family": "a"}] * 3 + [{"task": "b", "family": "b"}]
        y = np.array([[1, 0]] * 3 + [[0, 1]])
        cost = np.tile([1, 4], (4, 1))
        result = comparison.paired_differences(rows, y, cost, np.ones(4, bool), np.zeros(4, bool))
        self.assertEqual(result["accuracy_pp"]["estimate"], 50)
        self.assertEqual(result["accuracy_pp"]["ci95"]["lower"], 50)
        self.assertEqual(result["accuracy_pp"]["ci95"]["upper"], 50)
        with self.assertRaisesRegex(ValueError, "always-Sol cost"):
            comparison.paired_differences(rows, y, np.zeros((4, 2)), np.ones(4, bool), np.zeros(4, bool))


class ComparisonProvenanceTests(unittest.TestCase):
    def test_calibration_freezes_renamed_methods_without_reading_final_grades(self):
        with tempfile.TemporaryDirectory() as temporary:
            data, native, output, source = fixture(Path(temporary))
            read, reads = core.read, []

            def guarded(path):
                reads.append(str(path))
                self.assertNotEqual(Path(path).name, "grades-final.json")
                return read(path)

            with patch.object(workflow, "ROOT", source), patch.object(workflow, "checked_data", return_value=data), \
                 patch.object(core, "read", side_effect=guarded):
                comparison.calibrate(data, native, output)
            frozen = read(output / "comparison-freeze.json")
            self.assertFalse(frozen["new_blind_evaluation"])
            self.assertFalse(frozen["final_outcomes_read_by_comparison"])
            self.assertEqual(len(frozen["choices"]), 8)
            prior = read(data / "reference/calibration-frontiers.json")
            current = read(output / "calibration-frontiers.json")
            for method, value in prior["methods"].items():
                self.assertEqual(current["methods"][method], value)
            self.assertEqual(set(current["methods"]) - set(prior["methods"]), set(comparison.METHODS))
            binding = read(output / "combined-model.json")
            self.assertEqual(binding["methods"], ["fast_lora"])
            self.assertEqual(binding["quality_model_sha256"], core.sha(native / "native/trained.json"))
            self.assertEqual(read(output / "cost-calibration.json")["model_sha256"], core.sha(output / "combined-model.json"))
            published_choices = read(data / "metadata/presentation-policies.json")["choices"]
            for choice in frozen["choices"]:
                self.assertEqual(choice["reference"], published_choices[choice["reference_method"]][str(choice["cap_fraction"])])
            self.assertTrue(any(path.endswith("scores-final.json") for path in reads))

    def test_changed_grade_source_runtime_or_score_model_is_rejected(self):
        for changed in ("grade", "source", "runtime", "score"):
            with self.subTest(changed=changed), tempfile.TemporaryDirectory() as temporary:
                data, native, _, source = fixture(Path(temporary))
                if changed == "grade":
                    path = data / "grades/grades-train.json"
                    value = core.read(path)
                    value["rows"][0]["cheap_correct"] = 1 - value["rows"][0]["cheap_correct"]
                    replace_json(path, value)
                elif changed == "source":
                    (source / "native/fast_runtime.py").write_text("# Changed implementation\n", encoding="utf-8")
                elif changed == "runtime":
                    replace_json(native / "native/fast-runtime.json", {"changed": True})
                else:
                    path = native / "native/scores-final.json"
                    value = core.read(path)
                    value["model_sha256"] = "0" * 64
                    replace_json(path, value)
                with patch.object(workflow, "ROOT", source), self.assertRaises(ValueError):
                    comparison.checked_native(data, native)

    def test_existing_directory_is_preserved_and_source_drift_blocks_final_replay(self):
        with tempfile.TemporaryDirectory() as temporary:
            data, native, output, source = fixture(Path(temporary))
            with patch.object(workflow, "ROOT", source), patch.object(workflow, "checked_data", return_value=data):
                comparison.calibrate(data, native, output)
                before = {p.name: p.read_bytes() for p in output.iterdir() if p.is_file()}
                with self.assertRaisesRegex(ValueError, "new comparison directory"):
                    comparison.calibrate(data, native, output)
                self.assertEqual(before, {p.name: p.read_bytes() for p in output.iterdir() if p.is_file()})
                path = output / "comparison-freeze.json"
                declaration = core.read(path)
                declaration["comparison_source_sha256"] = "0" * 64
                replace_json(path, declaration)
                with patch.object(frontier, "evaluate") as evaluated, self.assertRaisesRegex(ValueError, "Frozen comparison"):
                    comparison.evaluate(data, output)
                evaluated.assert_not_called()

    def test_synthetic_replay_keeps_choices_and_zero_paired_difference_for_identical_model(self):
        with tempfile.TemporaryDirectory() as temporary:
            data, native, output, source = fixture(Path(temporary))
            with patch.object(workflow, "ROOT", source), patch.object(workflow, "checked_data", return_value=data), \
                 patch.object(workflow, "invoke"):
                comparison.calibrate(data, native, output)
                saved = (output / "comparison-freeze.json").read_bytes()
                comparison.evaluate(data, output)
                self.assertEqual(saved, (output / "comparison-freeze.json").read_bytes())
                result = core.read(output / "comparison-report.json")
                self.assertEqual(result["n"], 2)
                self.assertFalse(result["new_blind_evaluation"])
                for row in result["comparisons"]:
                    if not row.get("infeasible"):
                        for value in row["candidate_minus_reference"].values():
                            self.assertEqual(value["estimate"], 0)
                            self.assertEqual(value["ci95"]["lower"], 0)
                            self.assertEqual(value["ci95"]["upper"], 0)
                report = (output / "comparison-report.json").read_bytes()
                with self.assertRaises(FileExistsError):
                    comparison.evaluate(data, output)
                self.assertEqual(report, (output / "comparison-report.json").read_bytes())


if __name__ == "__main__":
    unittest.main()
