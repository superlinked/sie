import json
import contextlib
import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import router


class PortableRouterTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)

    def tearDown(self):
        self.temporary.cleanup()

    def write(self, name, value):
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value))
        return path

    def recorded(self):
        metadata = [{"id": "a", "source": "mmlupro", "task": "biology", "family": "family-a"}]
        bundle = {
            "source_plan_sha256": "1" * 64,
            "pair_spec_sha256": "2" * 64,
            "cost_scope": "observed USD",
            "acquisition_protocol_sha256": "3" * 64,
            "grader_sha256": "4" * 64,
            "source_rows": metadata,
            "frozen_before_final_model_outcomes": True,
        }
        frozen = self.write("routes.json", bundle)
        grades = {
            "format": "routerbench-paired-grades-v1",
            "split": "final",
            "source_plan_sha256": bundle["source_plan_sha256"],
            "pair_spec_sha256": bundle["pair_spec_sha256"],
            "cost_scope": bundle["cost_scope"],
            "parent_acquisition_protocol_sha256": "3" * 64,
            "parent_grader_sha256": "4" * 64,
            "frozen_final_routing_sha256": router.core.sha(frozen),
            "acquisition_protocol_sha256": "5" * 64,
            "grader_sha256": "6" * 64,
            "rows": [{"id": "a", "dataset": "mmlupro", "task": "biology", "family": "family-a"}],
        }
        return frozen, self.write("grades.json", grades)

    def test_modified_routes_cannot_relabel_original_acquisition(self):
        frozen, grades = self.recorded()
        altered = router.core.read(frozen)
        altered["new_route"] = True
        frozen.write_text(json.dumps(altered))
        with patch.object(router.paired_report, "evaluate") as evaluate:
            with self.assertRaisesRegex(ValueError, "frozen parent lineage"):
                router.evaluate_recorded(frozen, grades, self.root / "report.json")
        evaluate.assert_not_called()
        self.assertFalse((self.root / "report.json").exists())

    def test_replay_compares_full_report_and_retains_mismatch(self):
        frozen, grades = self.recorded()
        output = self.root / "report.json"
        expected = self.write("expected.json", {"questions": 2})
        with patch.object(router.paired_report, "evaluate", return_value={"questions": 1}):
            with self.assertRaisesRegex(ValueError, "Recorded replay differs"):
                router.evaluate_recorded(frozen, grades, output, expected_report=expected)
        self.assertTrue(output.is_file())
        self.assertEqual(router.core.read(output)["questions"], 1)

    def test_replay_allows_only_small_float_roundoff(self):
        expected = {"accuracy": 93.5555555555555, "questions": 3043, "verified": True, "ids": ["a", "b"]}
        close = {**expected, "accuracy": 93.5555555555556}
        self.assertTrue(router.same_report(expected, close))
        for changed in (
            {**expected, "accuracy": 93.5556},
            {**expected, "questions": 3044},
            {**expected, "verified": 1},
            {**expected, "ids": ["b", "a"]},
            {**expected, "extra": None},
            {**expected, "accuracy": float("nan")},
        ):
            self.assertFalse(router.same_report(expected, changed))

    def test_training_adapter_accepts_only_train_grade_path(self):
        args = router.parser().parse_args(["fit-embedding", "--data", str(self.root), "--run", str(self.root / "new")])
        with patch.object(router.core, "train_embedding", return_value={"fit": {"penalty": 0.1}}) as fit:
            with patch.object(router.core, "sha", return_value="7" * 64):
                router.execute(args)
        fit.assert_called_once_with(
            self.root / "new", self.root / "recordings/grades-train.json", self.root / "features"
        )

    def test_prepare_uses_reference_metadata_without_outcomes(self):
        args = router.parser().parse_args(["prepare", "--data", str(self.root), "--run", str(self.root / "new")])
        with patch.object(router.core, "checked_plan", return_value=({"acquisition_protocol_sha256": "8" * 64}, [])):
            with patch.object(router.core, "prepare", return_value={"populations": {"train": ["a"]}}) as prepare:
                with patch.object(router.core, "sha", return_value="9" * 64):
                    router.execute(args)
        prepare.assert_called_once_with(self.root / "source", self.root / "pair.json", self.root / "new", "8" * 64)

    def test_regenerated_embeddings_are_used_for_training_and_scoring(self):
        features = self.root / "new-features"
        common = ["--data", str(self.root), "--run", str(self.root / "new"), "--features", str(features)]
        with patch.object(router.core, "sha", return_value="7" * 64):
            with patch.object(router.core, "train_embedding", return_value={"fit": {"penalty": 1}}) as fit:
                router.execute(router.parser().parse_args(["fit-embedding", *common]))
            with patch.object(router.core, "score_embedding") as score:
                router.execute(router.parser().parse_args(["score", *common, "--method", "embedding"]))
        fit.assert_called_once_with(self.root / "new", self.root / "recordings/grades-train.json", features)
        score.assert_called_once_with(self.root / "new", features, self.root / "new/embedding-scores.json")
        with patch.object(router.paired_native, "score_native") as native:
            with self.assertRaisesRegex(ValueError, "applies only"):
                router.execute(router.parser().parse_args(["score", *common, "--method", "native"]))
        native.assert_not_called()

    def test_choice_answer_requires_public_option_labels_before_loading_service(self):
        args = router.parser().parse_args(
            [
                "answer",
                "--run",
                str(self.root),
                "--method",
                "embedding",
                "--target-gap-pp",
                "0",
                "--question",
                "2+2?",
                "--answer-type",
                "choice",
                "--output",
                str(self.root / "answer"),
            ]
        )
        with self.assertRaisesRegex(ValueError, "Supply --choices"):
            router.execute(args)

    def test_answer_requires_an_explicit_operating_point(self):
        common = ["answer", "--run", "run", "--method", "embedding", "--question", "Example?", "--output", "answer"]
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                router.parser().parse_args(common)
            with self.assertRaises(SystemExit):
                router.parser().parse_args([*common, "--candidate-id", "7", "--target-gap-pp", "1"])
        args = router.parser().parse_args([*common, "--candidate-id", "7"])
        self.assertEqual(args.candidate_id, 7)
        self.assertIsNone(args.target_gap_pp)

    def test_show_defaults_to_every_recorded_operating_point(self):
        metric = {"quality_gap_from_always_strong_pp": 0.5, "total_usd_per_10000": 9.0,
                  "net_savings_percent": 35.0, "answer_accuracy_percent": 92.0, "answer_model_usd_per_10000": 8.8}
        row = {"metrics": metric, "confidence95": {"quality_gap_from_always_strong_pp": {"lower": 0.1, "upper": 0.9}}}
        names = ["always_strong", "always_cheap"] + [f"{method}@{target}pp"
                 for method in ("base", "lora", "embedding", "topic", "random") for target in (0, 1, 2, 5)]
        report = self.write("report.json", {"format": "routerbench-paired-final-report-v1", "questions": 3,
                                           "policies": {name: row for name in names}})
        args = router.parser().parse_args(["show", "--report", str(report)])
        self.assertIsNone(args.target_gap_pp)
        text = router.execute(args)
        self.assertEqual(text.count("| Embedding + classifier |"), 4)
        self.assertIn("GPT-6.1 Sol and GPT-6 Luna", text)
        self.assertEqual(router.show_report(report, 2).count("| Embedding + classifier |"), 1)

    def test_frozen_portable_sources_match_manifest(self):
        directory = Path(router.__file__).parent
        manifest = router.core.read(directory / "frozen-source-manifest.json")
        self.assertEqual(len(manifest["files_sha256"]), 6)
        for name, digest in manifest["files_sha256"].items():
            self.assertEqual(router.core.sha(directory / name), digest)
            source = (directory / name).read_text()
            self.assertNotIn("/private/", source)
            self.assertNotIn("/Users/", source)
            self.assertNotIn("import modal", source)


if __name__ == "__main__":
    unittest.main()
