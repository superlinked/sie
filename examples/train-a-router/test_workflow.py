"""Tiny independent public-reader checks; no provider, model or real outcome calls."""

import ast
import copy
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

import calibration_view
import cost_reader
from embedding import recipes as core
from embedding.cost_iteration import study
import live
import openai_population
import workflow


def replace_json(path, value):
    if path.exists():
        path.unlink()
    core.write(path, value)


def make_run(root):
    """Construct genuine hash-bound tiny recipe/grade files, not published outcomes."""
    run = root / "run"
    run.mkdir()
    pair = {"models": {"cheap": "gpt-6-luna", "strong": "gpt-6.1-sol"},
            "cost_scope": "Synthetic known selected-response USD"}
    core.write(run / "pair.json", pair)
    plan = {"format": "bbeh-recipe-plan-v1", "protocol": core.PROTOCOL,
            "source_plan_sha256": "a" * 64, "pair_sha256": core.sha(run / "pair.json"),
            "pair": pair, "recipe_source_sha256": core.sha(core.__file__), "roles": {}}
    rows, paths, dollars = {}, {}, {}
    for role, n in (("train", 4), ("development", 3), ("calibration", 2), ("final", 2)):
        rows[role] = [{"id": f"{role}-{i}", "role": role, "task": "synthetic",
                       "family": f"family-{role}-{i}", "dataset": "bbeh",
                       "request": f"Shared arithmetic question repeat {i} plus {i + 1}",
                       "response_contract": {"type": "exact_string", "max_length": 2048}}
                      for i in range(n)]
        path = run / f"rows-{role}.json"
        core.write(path, rows[role])
        plan["roles"][role] = {"n": n, "rows_sha256": core.sha(path), "inputs_sha256": "c" * 64}
        if role in {"train", "development"}:
            dollars[role] = np.array([[0.01 + i * 0.003, 0.08 + i * 0.01] for i in range(n)])
            paths[role] = root / f"grades-{role}.json"
            grade = {"format": "bbeh-paired-grades-v1", "role": role,
                     "source_plan_sha256": plan["source_plan_sha256"], "pair_sha256": plan["pair_sha256"],
                     "models": pair["models"], "cost_scope": pair["cost_scope"],
                     "grader_sha256": "d" * 64, "acquisition_protocol_sha256": "e" * 64,
                     "rows": [{"id": row["id"], "cheap_correct": i % 2,
                               "strong_correct": (i + 1) % 2, "cheap_record_sha256": "f" * 64,
                               "strong_record_sha256": "0" * 64,
                               "cheap_cost_usd": str(dollars[role][i, 0]),
                               "strong_cost_usd": str(dollars[role][i, 1])}
                              for i, row in enumerate(rows[role])]}
            core.write(paths[role], grade)
    core.write(run / "plan.json", plan)
    quality = {"plan_sha256": core.sha(run / "plan.json"),
               "train_grades_sha256": core.sha(paths["train"]),
               "development_grades_sha256": core.sha(paths["development"]),
               "calibration_or_final_outcomes_accessed": False,
               "different_student_fit_metadata": "This intentionally is not the original model hash"}
    core.write(run / "model.json", quality)
    for role in ("calibration", "final"):
        core.write(run / f"scores-{role}.json", {
            "format": "bbeh-input-only-scores-v1", "role": role,
            "ids": [r["id"] for r in rows[role]], "source_plan_sha256": plan["source_plan_sha256"],
            "pair_sha256": plan["pair_sha256"], "model_sha256": core.sha(run / "model.json"),
            "outcomes_accessed": False,
            "methods": {"embedding": {"probabilities": [[0.2, 0.8], [0.8, 0.4]],
                         "expected_usd": [[0.01, 0.08], [0.02, 0.09]],
                         "router_kind": "embedding", "metadata_assisted": False}}})
    return run, plan, rows, paths, dollars


class CostReaderTests(unittest.TestCase):
    def test_student_model_hash_and_math_parity_without_future_grades(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run, _, rows, paths, dollars = make_run(root)
            old = {role: core.read(run / f"scores-{role}.json") for role in ("calibration", "final")}
            actual_read, reads = core.read, []

            def recorded_read(path):
                reads.append(str(path))
                self.assertNotIn("grades-calibration", str(path))
                self.assertNotIn("grades-final", str(path))
                return actual_read(path)

            with patch.object(core, "read", side_effect=recorded_read):
                receipt = cost_reader.fit_score(run, paths["train"], paths["development"], root / "cost")
            self.assertEqual(receipt["fit_n"], 7)
            self.assertEqual(receipt["alpha"], 0.1)
            model = actual_read(root / "cost/cost-model.json")
            self.assertTrue(model["reproduction"])
            self.assertFalse(model["new_blind_evaluation"])
            self.assertEqual(model["quality_model_sha256"], core.sha(run / "model.json"))
            _, matrix, text = core.text_matrix(rows["train"] + rows["development"])
            expected = study.fit_ridge(matrix, np.concatenate((dollars["train"], dollars["development"])), 0.1)
            self.assertEqual(text, model["text_features"])
            for name in ("weights", "bias", "train_residual_smearing"):
                np.testing.assert_allclose(model["models"]["text_length_log_ridge"][name], expected[name],
                                           rtol=1e-12, atol=1e-12)
            for role in ("calibration", "final"):
                prediction = actual_read(root / f"cost/costs-{role}.json")
                augmented = actual_read(root / f"cost/scores-{role}.json")
                self.assertEqual(augmented["methods"]["embedding"], old[role]["methods"]["embedding"])
                np.testing.assert_allclose(prediction["expected_usd"],
                    study.predict_rows(model, rows[role], "text_length_log_ridge"), rtol=1e-12, atol=1e-12)
                self.assertEqual(augmented["methods"]["embedding__textcost"]["expected_usd"], prediction["expected_usd"])
            self.assertTrue(any("rows-final.json" in path for path in reads))

    def test_changed_grade_binding_is_rejected_before_fit(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run, _, _, paths, _ = make_run(root)
            grade = core.read(paths["train"])
            grade["rows"][0]["cheap_cost_usd"] = "0.012"
            replace_json(paths["train"], grade)
            with patch.object(study, "fit_ridge") as fitted, self.assertRaisesRegex(ValueError, "lineage"):
                cost_reader.fit_score(run, paths["train"], paths["development"], root / "cost")
            fitted.assert_not_called()

    def test_future_score_wrong_role_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run, _, _, paths, _ = make_run(root)
            score = core.read(run / "scores-calibration.json")
            score["role"] = "final"
            replace_json(run / "scores-calibration.json", score)
            with self.assertRaisesRegex(ValueError, "source-role-order"):
                cost_reader.fit_score(run, paths["train"], paths["development"], root / "cost")


class ReaderUiAndCallTests(unittest.TestCase):
    @unittest.skipUnless(shutil.which("node"), "Node is needed for actual generated-script parsing")
    def test_calibration_html_emits_valid_javascript(self):
        path = Path(calibration_view.__file__)
        tree = ast.parse(path.read_text())
        template = next(ast.literal_eval(node.value) for node in tree.body
                        if isinstance(node, ast.Assign) and any(
                            isinstance(target, ast.Name) and target.id == "TEMPLATE" for target in node.targets))
        script = template.split("<script>", 1)[1].split("</script>", 1)[0].replace("__DATA__", "[]")
        result = subprocess.run(["node", "--check", "-"], input=script, text=True, capture_output=True, check=False)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_wrong_returned_model_or_tier_not_presented_as_success_and_never_retried(self):
        for resolved_model, tier in (("not-the-selected-model", "default"), ("gpt-6-luna", "flex")):
            with self.subTest(model=resolved_model, tier=tier), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                response = {"status": "complete", "terminal_response": True, "model": "gpt-6-luna",
                            "resolved_model": resolved_model, "service_tier": tier, "answer": "synthetic",
                            "valid_answer": True}
                with patch.dict(os.environ, {"OPENAI_API_KEY": "synthetic-unused"}), \
                    patch.object(live, "selected_policy", return_value=({"mode": "always_cheap"}, "a" * 64, "b" * 64)), \
                    patch.object(openai_population, "acquire", return_value=response) as acquire:
                    try:
                        result = live.answer(root / "run", "Synthetic question", root / "answer",
                                             method="embedding", candidate_id=0)
                    except ValueError:
                        pass
                    else:
                        self.assertNotEqual(result["status"], "complete")
                acquire.assert_called_once()
                self.assertEqual(core.read(root / "answer/answer.json"), response)

    def test_replay_numeric_tolerance_keeps_structure_and_large_drift_strict(self):
        expected = {"quality": [0.123456789], "count": 7, "nested": {"ok": True}}
        rounded = copy.deepcopy(expected)
        rounded["quality"][0] += 1e-12
        self.assertTrue(workflow.same_values(expected, rounded))
        rounded["quality"][0] += 1e-5
        self.assertFalse(workflow.same_values(expected, rounded))
        self.assertFalse(workflow.same_values(expected, {**expected, "extra": 1}))
        self.assertFalse(workflow.same_values(expected, {**expected, "count": 7.0}))


class CalibrationSelectionTests(unittest.TestCase):
    def fixture(self, root):
        embedding, plan, _, _, _ = make_run(root)
        run = root / "student"
        run.mkdir()
        shutil.move(embedding, run / "embedding")
        candidate = {"candidate_id": 2, "policy": {"mode": "always_cheap"}, "calibration": {}}
        cal = {"format": "bbeh-calibration-frontiers-v1", "final_outcomes_accessed": False,
               "plan_sha256": core.sha(run / "embedding/plan.json"),
               "source_plan_sha256": plan["source_plan_sha256"], "pair_sha256": plan["pair_sha256"],
               "methods": {"embedding": {"model_sha256": "b" * 64, "metadata_assisted": False,
                                          "candidates": [candidate], "frontier": [candidate]}}}
        core.write(run / "internal/calibration-frontiers.json", cal)
        frozen = {key: cal[key] for key in ("plan_sha256", "source_plan_sha256", "pair_sha256")}
        frozen.update(calibration_sha256=core.sha(run / "internal/calibration-frontiers.json"),
                      methods={"embedding": {"model_sha256": "b" * 64, "candidates": [candidate]}})
        core.write(run / "internal/frozen-frontiers.json", frozen)
        core.write(run / "reproduction.json", {"artifact_directory": str(root / "data")})
        self.bind_envelope(run)
        return run

    def bind_envelope(self, run):
        replace_json(run / "calibration.json", {
            "new_blind_evaluation": False,
            "reproduction_sha256": core.sha(run / "reproduction.json"),
            "calibration_sha256": core.sha(run / "internal/calibration-frontiers.json"),
            "frozen_sha256": core.sha(run / "internal/frozen-frontiers.json")})

    def test_own_calibration_and_stale_envelope(self):
        with tempfile.TemporaryDirectory() as temporary:
            run = self.fixture(Path(temporary))
            with patch.object(workflow, "checked_run") as checked:
                policy, model_sha, _ = live.selected_policy(run, "embedding", 2)
                self.assertEqual(policy, {"mode": "always_cheap"})
                self.assertEqual(model_sha, "b" * 64)
                checked.assert_called_once_with(str(Path(temporary) / "data"), run.resolve())
                cal = core.read(run / "internal/calibration-frontiers.json")
                cal["pair_sha256"] = "f" * 64
                replace_json(run / "internal/calibration-frontiers.json", cal)
                with self.assertRaisesRegex(ValueError, "Reproduction calibration changed"):
                    live.selected_policy(run, "embedding", 2)

    def test_rebound_envelope_cannot_hide_pair_mismatch(self):
        with tempfile.TemporaryDirectory() as temporary:
            run = self.fixture(Path(temporary))
            frozen = core.read(run / "internal/frozen-frontiers.json")
            frozen["pair_sha256"] = "f" * 64
            replace_json(run / "internal/frozen-frontiers.json", frozen)
            self.bind_envelope(run)
            with patch.object(workflow, "checked_run"), self.assertRaisesRegex(ValueError, "source/pair/plan"):
                live.selected_policy(run, "embedding", 2)

    def test_unknown_or_metadata_only_choice_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            run = self.fixture(Path(temporary))
            with patch.object(workflow, "checked_run"), self.assertRaisesRegex(ValueError, "Unknown candidate"):
                live.selected_policy(run, "embedding", 99)
            cal = core.read(run / "internal/calibration-frontiers.json")
            cal["methods"]["embedding"]["metadata_assisted"] = True
            replace_json(run / "internal/calibration-frontiers.json", cal)
            frozen = core.read(run / "internal/frozen-frontiers.json")
            frozen["calibration_sha256"] = core.sha(run / "internal/calibration-frontiers.json")
            replace_json(run / "internal/frozen-frontiers.json", frozen)
            self.bind_envelope(run)
            with patch.object(workflow, "checked_run"), self.assertRaisesRegex(ValueError, "benchmark metadata"):
                live.selected_policy(run, "embedding", 2)


class WorkflowIntegrationTests(unittest.TestCase):
    def test_student_cost_calibration_and_live_binding_use_own_model_and_no_final_answers(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            embedding, plan, rows, grade_paths, _ = make_run(root)
            run, data = root / "student", root / "data"
            run.mkdir()
            shutil.move(embedding, run / "embedding")
            data.mkdir()
            core.write(data / "manifest.json", {"synthetic": True})
            core.write(data / "reference/frozen-frontiers.json", {"synthetic_reference": True})
            for role, path in grade_paths.items():
                (data / "grades").mkdir(exist_ok=True)
                shutil.copyfile(path, data / f"grades/grades-{role}.json")
            calibration_grade = core.read(grade_paths["train"])
            calibration_grade["role"] = "calibration"
            calibration_grade["rows"] = [
                {**calibration_grade["rows"][i], "id": row["id"]}
                for i, row in enumerate(rows["calibration"])]
            core.write(data / "grades/grades-calibration.json", calibration_grade)
            core.write(data / "fees.json", {
                "source_plan_sha256": plan["source_plan_sha256"], "model": core.MODEL,
                "hf_revision": core.REVISION, "instruction": core.INSTRUCTION,
                "cost_basis": "validated_singleton_projection",
                "rows": [{"id": row["id"], "projected_singleton_usd": "0.00001"}
                         for role in ("calibration", "final") for row in rows[role]]})
            core.write(run / "reproduction.json", {
                "format": "bbeh-public-reproduction-v1", "artifact_directory": str(data),
                "manifest_sha256": core.sha(data / "manifest.json"),
                "reference_frozen_sha256": core.sha(data / "reference/frozen-frontiers.json"),
                "new_blind_evaluation": False})
            for role in ("calibration", "final"):
                native = core.read(run / f"embedding/scores-{role}.json")
                native["model_sha256"] = "f" * 64
                native["methods"] = {name: {**copy.deepcopy(native["methods"]["embedding"]),
                                                   "router_kind": "native"} for name in ("base", "lora")}
                core.write(run / f"scores/native-{role}.json", native)
            actual_read = core.read

            def no_final_grades(path):
                self.assertNotIn("grades-final", str(path))
                return actual_read(path)

            with patch.object(workflow, "checked_data", side_effect=lambda path: Path(path).resolve()), \
                    patch.object(core, "read", side_effect=no_final_grades):
                cost = workflow.fit_cost(data, run)
                self.assertEqual(cost["alpha"], 0.1)
                receipt = workflow.calibrate(data, run)
                self.assertEqual(receipt["new_model_calls"], 0)
                calibration = actual_read(run / "internal/calibration-frontiers.json")
                method = calibration["methods"]["embedding__textcost"]
                candidate = method["frontier"][0]
                chosen, model_sha, _ = live.selected_policy(run, "embedding__textcost", candidate["candidate_id"])
                self.assertEqual(chosen, candidate["policy"])
                quality, model = live.binding(run, "embedding__textcost", model_sha)
                self.assertEqual(quality, "embedding")
                self.assertEqual(model, actual_read(run / "embedding/model.json"))
            self.assertFalse(receipt["new_model_calls"])
            self.assertFalse(actual_read(run / "calibration.json")["new_blind_evaluation"])
            self.assertFalse((data / "grades/grades-final.json").exists())
            frozen = actual_read(run / "internal/frozen-frontiers.json")
            self.assertEqual(set(frozen["methods"]),
                             {"embedding", "embedding__textcost", "base", "lora", "base__textcost", "lora__textcost"})
            self.assertEqual(frozen["ids"], [row["id"] for row in rows["final"]])

if __name__ == "__main__":
    unittest.main()
