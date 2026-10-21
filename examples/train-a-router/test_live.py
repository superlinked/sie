"""Live routing boundaries, using invented model replies and no network."""

import json
import unittest
from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import MagicMock, patch

import numpy as np

import live


class CandidateSelectionTests(unittest.TestCase):
    """Exercise live selection against tiny saved artifacts, never real services."""

    def setUp(self):
        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.run = Path(temporary.name) / "run"
        self.output = Path(temporary.name) / "answer"
        self.plan = {
            "source_plan_sha256": "c" * 64,
            "pair_spec_sha256": "d" * 64,
            "acquisition_protocol_sha256": "e" * 64,
            "populations": {"calibration": ["invented-calibration-row"]},
            "pair": {
                "cost_scope": "Invented USD fixture",
                "arms": {
                    "cheap": {"arm": "luna", "model": "gpt-6-luna"},
                    "strong": {"arm": "sol", "model": "gpt-6.1-sol"},
                }
            }
        }
        live.core.write(self.run / "plan.json", self.plan)
        lineage = {
            "plan_sha256": live.core.sha(self.run / "plan.json"),
            "training_grades_sha256": "a" * 64,
        }
        costs = {
            "unit": "USD",
            "arm_order": ["cheap", "strong"],
            "class_order": [0, 1],
            "means": [[0.001, 0.001], [0.02, 0.02]],
        }
        self.models = {
            "embedding": {
                **lineage,
                "fit": {
                    "models": {
                        "cheap": {"weights": [0.0, 0.0], "bias": float(np.log(0.25))},
                        "strong": {"weights": [0.0, 0.0], "bias": float(np.log(4.0))},
                    },
                    "conditional_cost_usd": costs,
                },
            },
            "native": {**lineage, "conditional_cost_usd": costs},
        }
        self.model_paths = {
            "embedding": self.run / "embedding.json",
            "native": self.run / "native/trained.json",
        }
        for family, path in self.model_paths.items():
            live.core.write(path, self.models[family])
        cheap = {"format": live.utility_router.FORMAT, "mode": "always_cheap"}
        strong = {"format": live.utility_router.FORMAT, "mode": "always_strong"}
        utility = {
            "format": live.utility_router.FORMAT,
            "mode": "utility",
            "lambda": {"numerator": "0", "denominator": "1"},
        }
        self.calibration = {
            "format": "routerbench-paired-calibration-v1",
            "protocol": live.frontier_analysis.paired_calibration.PROTOCOL,
            "ids": self.plan["populations"]["calibration"],
            "methods": {
                method: {
                    "protocol": live.utility_router.PROTOCOL,
                    "n": 1,
                    "candidates": [
                        {"candidate_id": index, "policy": deepcopy(policy)}
                        for index, policy in enumerate([cheap, strong, *([utility] * 6)])
                    ],
                    "selected": {"1": {"candidate_id": 1, "policy": strong}},
                }
                for method in ("embedding", "base", "lora", "topic", "topic_metadata_diagnostic")
            }
        }
        self.calibration["methods"]["embedding"]["candidates"][7]["policy"]["lambda"]["numerator"] = "100"
        self.calibration["methods"]["base"]["candidates"][7]["policy"] = {
            "format": live.utility_router.FORMAT,
            "mode": "cost_limit",
        }
        self.calibration["methods"]["lora"]["candidates"][7]["policy"]["lambda"]["numerator"] = "200"
        self.bundle = {
            **lineage,
            **{key: self.plan[key] for key in (
                "source_plan_sha256", "pair_spec_sha256", "acquisition_protocol_sha256",
            )},
            "cost_scope": self.plan["pair"]["cost_scope"],
            "grader_sha256": "f" * 64,
            "calibration_grades_sha256": "1" * 64,
            "fee_projection_sha256": "2" * 64,
            "calibration_source_sha256": live.core.sha(live.frontier_analysis.paired_calibration.__file__),
            "utility_source_sha256": live.core.sha(live.utility_router.__file__),
            "scores_sha256": {},
            "policies": {
                f"{method}@1pp": {"rule": strong, "calibration_candidate_id": 1}
                for method in self.calibration["methods"]
            },
        }
        for family in self.models:
            self.seal_scores(family)
        self.calibration.update({key: deepcopy(self.bundle[key]) for key in live.frontier_analysis.PROVENANCE_KEYS})
        self.calibration_path = self.run / "frozen/calibration.json"
        live.core.write(self.calibration_path, self.calibration)
        self.bundle["calibration_sha256"] = live.core.sha(self.calibration_path)
        self.enterContext(patch.object(live.core, "checked_plan", return_value=(self.plan, [])))
        self.routes_loader = self.enterContext(
            patch.object(live.reproduction, "checked_routes", return_value=(self.bundle, None))
        )
        self.real_calibration_loader = live.frontier_analysis.load_calibration
        self.calibration_loader = self.enterContext(
            patch.object(live.frontier_analysis, "load_calibration", return_value=(self.bundle, self.calibration))
        )
        self.sie = self.enterContext(patch.object(live, "sie_client", side_effect=AssertionError("Unexpected SIE access")))
        self.http = self.enterContext(
            patch.object(live.httpx, "Client", side_effect=AssertionError("Unexpected HTTP access"))
        )

    def seal_scores(self, family):
        key = "model_sha256" if family == "embedding" else "trained_sha256"
        path = self.run / f"{family}-scores.json"
        path.write_text(json.dumps({key: live.core.sha(self.model_paths[family])}))
        self.bundle["scores_sha256"][family] = live.core.sha(path)

    def assert_rejected_before_api(self, message, *, error=ValueError, method="embedding", **selection):
        with self.assertRaisesRegex(error, message):
            live.answer(self.run, "Invented question", self.output, method=method, **selection)
        self.sie.assert_not_called()
        self.http.assert_not_called()
        self.assertFalse(self.output.exists(), "Invalid selection must not create an execution intent")

    def test_arbitrary_candidate_uses_its_method_and_preserves_explicit_legacy_selection(self):
        for method in ("embedding", "base", "lora"):
            with self.subTest(method=method):
                plan, rule, model = live.checked_model(self.run, method, candidate_id=7)
                self.assertEqual(plan, self.plan)
                self.assertEqual(rule, self.calibration["methods"][method]["candidates"][7]["policy"])
                self.assertEqual(model, self.models["embedding" if method == "embedding" else "native"])
                self.calibration_loader.assert_called_with(self.run)
        self.calibration_loader.reset_mock()
        _, rule, _ = live.checked_model(self.run, "embedding", target=1)
        self.assertEqual(rule["mode"], "always_strong")
        self.calibration_loader.assert_not_called()

    def test_candidate_zero_and_one_bypass_router_and_persist_the_selected_identity(self):
        for method in ("embedding", "base", "lora"):
            for candidate_id, arm in ((0, "cheap"), (1, "strong")):
                with self.subTest(method=method, candidate_id=candidate_id):
                    output = self.output.with_name(f"{method}-{candidate_id}")
                    with (
                        patch.object(live, "embed") as embed,
                        patch.object(live.paired_native, "checkpoint_probabilities") as native,
                        patch.object(live, "solve", return_value={
                            "answer": "A", "valid_answer": True, "quality": "Not graded",
                        }) as solve,
                    ):
                        result = live.answer(
                            self.run, "Invented question", output, method=method, candidate_id=candidate_id
                        )
                    solve.assert_called_once()
                    self.assertEqual(solve.call_args.args[1], arm)
                    self.assertEqual(result["model"], self.plan["pair"]["arms"][arm]["model"])
                    intent, routing = live.core.read(output / "intent.json"), live.core.read(output / "routing.json")
                    self.assertEqual(intent["calibration_candidate_id"], candidate_id)
                    self.assertIsNone(intent["target_gap_pp"])
                    self.assertEqual(routing["calibration_candidate_id"], candidate_id)
                    self.assertIsNone(routing["calibration_target_gap_pp"])
                    embed.assert_not_called()
                    native.assert_not_called()
        self.sie.assert_not_called()
        self.http.assert_not_called()

    def test_arbitrary_utility_candidates_change_the_arm_for_identical_predictions(self):
        routes = []
        with patch.object(live, "embed", return_value=(np.array([1.0, 0.0]), {"observed_credits": 1})) as embed:
            for candidate_id in (6, 7):
                result, _ = live.route(
                    self.run, "Same invented question", "embedding", candidate_id=candidate_id, client=MagicMock()
                )
                routes.append(result)
        self.assertEqual([row["arm"] for row in routes], ["strong", "cheap"])
        self.assertEqual([row["calibration_candidate_id"] for row in routes], [6, 7])
        self.assertEqual(routes[0]["predicted_correctness"], routes[1]["predicted_correctness"])
        np.testing.assert_allclose(routes[0]["predicted_correctness"], [0.2, 0.8])
        self.assertEqual(routes[0]["predicted_answer_usd"], routes[1]["predicted_answer_usd"])
        self.assertEqual(embed.call_count, 2)

    def test_missing_ambiguous_and_invalid_choices_are_rejected_before_api(self):
        cases = [({}, "Choose"), ({"target": 1, "candidate_id": 0}, "Choose")]
        cases.extend(({"target": value}, "Unknown recorded") for value in (-1, 3, True, 1.0, "1"))
        cases.extend(({'candidate_id': value}, "Unknown calibration candidate") for value in (-1, 8, True, 1.0, "1"))
        for selection, message in cases:
            with self.subTest(selection=selection):
                self.assert_rejected_before_api(message, **selection)
        self.assert_rejected_before_api("Unknown routing method", method="topic", candidate_id=0)

    def test_changed_plan_scores_or_model_bytes_are_rejected_before_api(self):
        cases = [(self.run / "plan.json", "embedding", "another run")]
        for family, method in (("embedding", "embedding"), ("native", "lora")):
            cases.extend([
                (self.run / f"{family}-scores.json", method, "Saved input scores changed"),
                (self.model_paths[family], method, "Saved router model changed"),
            ])
        for path, method, message in cases:
            with self.subTest(path=path.name, method=method):
                original = path.read_bytes()
                path.write_bytes(original + b"\n")
                try:
                    self.assert_rejected_before_api(message, method=method, candidate_id=7)
                finally:
                    path.write_bytes(original)

    def test_rehashed_model_from_another_training_run_is_rejected_before_api(self):
        for family, method in (("embedding", "embedding"), ("native", "base")):
            for field in ("plan_sha256", "training_grades_sha256"):
                with self.subTest(family=family, field=field):
                    changed = deepcopy(self.models[family])
                    changed[field] = "c" * 64
                    self.model_paths[family].write_text(json.dumps(changed))
                    self.seal_scores(family)
                    self.assert_rejected_before_api("Training lineage changed", method=method, candidate_id=7)

    def test_different_calibration_binding_is_rejected_before_api(self):
        self.calibration_loader.return_value = ({**self.bundle, "calibration_sha256": "d" * 64}, self.calibration)
        self.assert_rejected_before_api("Calibration binding changed", candidate_id=7)

    def test_missing_or_rejected_calibration_is_not_silently_replaced_by_a_recorded_policy(self):
        for error in (FileNotFoundError("calibration unavailable"), ValueError("calibration bytes changed")):
            with self.subTest(error=type(error).__name__):
                self.calibration_loader.side_effect = error
                self.assert_rejected_before_api("calibration", error=type(error), candidate_id=7)

    def test_candidate_identity_cannot_be_swapped_with_another_position(self):
        self.calibration["methods"]["embedding"]["candidates"][7]["candidate_id"] = 6
        self.assert_rejected_before_api("Calibration candidate identity changed", candidate_id=7)

    def test_live_candidate_works_with_original_and_retrospective_calibration_layouts(self):
        self.calibration_loader.side_effect = self.real_calibration_loader
        for retrospective in (False, True):
            with self.subTest(retrospective=retrospective):
                if retrospective:
                    internal = self.run / "frozen/internal/calibration.json"
                    internal.parent.mkdir()
                    self.calibration_path.replace(internal)
                    self.routes_loader.return_value = (self.bundle, {"evaluation_mode": "published-data reproduction"})
                _, rule, _ = live.checked_model(self.run, "embedding", candidate_id=7)
                self.assertEqual(rule, self.calibration["methods"]["embedding"]["candidates"][7]["policy"])
        self.sie.assert_not_called()
        self.http.assert_not_called()

    def test_actual_calibration_file_tampering_is_rejected_before_api(self):
        self.calibration_loader.side_effect = self.real_calibration_loader
        self.calibration_path.write_bytes(self.calibration_path.read_bytes() + b"\n")
        self.assert_rejected_before_api("Saved calibration file changed", candidate_id=7)

    def test_rehashed_calibration_from_another_model_is_rejected_before_api(self):
        self.calibration_loader.side_effect = self.real_calibration_loader
        self.calibration["scores_sha256"]["embedding"] = "9" * 64
        self.calibration_path.write_text(json.dumps(self.calibration))
        self.bundle["calibration_sha256"] = live.core.sha(self.calibration_path)
        self.assert_rejected_before_api("Calibration provenance differs: scores_sha256", candidate_id=7)


class LiveTests(unittest.TestCase):
    def test_self_hosted_and_unrelated_endpoints_do_not_receive_cloud_environment_key(self):
        with patch.dict(live.os.environ, {"SIE_API_KEY": "invented-cloud-key", "SIE_BASE_URL": "https://api.superlinked.com"}):
            with live.sie_client("http://localhost:8080", self_hosted=True) as local:
                self.assertFalse(local._api_key)
            with live.sie_client("https://unrelated.example.test") as unrelated:
                self.assertFalse(unrelated._api_key)
            with live.sie_client("https://api.superlinked.com") as cloud:
                self.assertEqual(cloud._api_key, "invented-cloud-key")

    def test_always_arm_bypasses_encoder_and_native_inference(self):
        plan = {"pair": {"arms": {"cheap": {"model": "cheap"}, "strong": {"model": "strong"}}}}
        for method in ("embedding", "lora", "base"):
            with (
                patch.object(live, "checked_model", return_value=(plan, {"mode": "always_cheap"}, {})),
                patch.object(live, "embed") as embed,
                patch.object(live.paired_native, "checkpoint_probabilities") as native,
            ):
                result, _ = live.route("unused", "invented question", method, 1)
            self.assertEqual(result["arm"], "cheap")
            self.assertIsNone(result["embedding"])
            embed.assert_not_called()
            native.assert_not_called()

    def test_live_encoder_preserves_instruction_dimensions_and_normalization(self):
        client = MagicMock()
        client.get_model.return_value = {"revision": live.core.REVISION, "dims": {"dense": 2560}}
        client.last_model_revision = "e" * 64
        client.encode.return_value = [
            {
                "id": "new",
                "model": live.core.MODEL,
                "dense": [2.0] * 2560,
                "request": {"id": "request-1", "credits_debited": 2},
            }
        ]
        vector, receipt = live.embed(client, "invented question", "new")
        self.assertEqual(vector.dtype, np.float32)
        self.assertAlmostEqual(float(np.linalg.norm(vector)), 1.0, places=6)
        self.assertEqual(receipt["observed_cloud_usd"], "0.00002")
        self.assertEqual(client.encode.call_args.kwargs["instruction"], live.core.INSTRUCTION)
        self.assertTrue(client.encode.call_args.kwargs["is_query"])
        client.get_model.return_value["revision"] = "different"
        with self.assertRaises(ValueError):
            live.embed(client, "invented question", "new")
        self.assertEqual(client.encode.call_count, 1)

    def test_selfhosted_encoder_does_not_claim_free_cloud_credits(self):
        client = MagicMock()
        client.get_model.return_value = {"revision": live.core.REVISION, "dims": {"dense": 2560}}
        client.encode.return_value = [{"id": "new", "model": live.core.MODEL, "dense": [1.0] * 2560}]
        _, receipt = live.embed(client, "invented question", "new", self_hosted=True)
        self.assertIsNone(receipt["observed_credits"])
        self.assertIsNone(receipt["observed_cloud_usd"])
        with self.assertRaises(ValueError):
            live.embed(client, "invented question", "new", self_hosted=False)

    def pair(self):
        return {
            "arms": {
                "cheap": {
                    "arm": "luna",
                    "model": "gpt-6-luna",
                    "settings": deepcopy(live.answer_models.OPENAI_SETTINGS),
                }
            },
            "request_builder_source_sha256": live.core.sha(live.answer_models.__file__),
            "prices_usd_per_million": deepcopy(live.answer_models.PRICES),
        }

    def item(self):
        return {"id": "invented", "request": "What is2+2?", "response_contract": {"type": "integer"}}

    def test_incomplete_answer_is_charged_once_and_not_retried(self):
        client = MagicMock()
        client.__enter__.return_value = client
        client.post.return_value.json.return_value = {
            "model": "gpt-6-luna",
            "service_tier": "default",
            "status": "incomplete",
            "output": [],
            "usage": {"input_tokens": 10, "output_tokens": 20, "output_tokens_details": {"reasoning_tokens": 15}},
        }
        with (
            patch.object(live.httpx, "Client", return_value=client),
            patch.dict(live.os.environ, {"OPENAI_API_KEY": "invented"}),
        ):
            result = live.solve(self.item(), "cheap", self.pair())
        self.assertEqual(client.post.call_count, 1)
        self.assertIsNone(result["answer"])
        self.assertFalse(result["valid_answer"])
        self.assertGreater(float(result["cost"]["usd_estimate"]), 0)

    def test_answer_setting_change_rejected_before_network(self):
        pair = self.pair()
        pair["arms"]["cheap"]["settings"]["max_output_tokens"] = 16
        with patch.object(live.httpx, "Client") as client:
            with self.assertRaises(ValueError):
                live.solve(self.item(), "cheap", pair)
            client.assert_not_called()


if __name__ == "__main__":
    unittest.main()
