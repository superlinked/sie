from __future__ import annotations

import importlib.util
import json
import sys
import unittest
from pathlib import Path
from typing import Any
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
RUN_SPEC = importlib.util.spec_from_file_location("rerank_run", ROOT / "run.py")
assert RUN_SPEC is not None
assert RUN_SPEC.loader is not None
run = importlib.util.module_from_spec(RUN_SPEC)
sys.modules[RUN_SPEC.name] = run
RUN_SPEC.loader.exec_module(run)


def strings(value: Any):
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for item in value.values():
            yield from strings(item)
    elif isinstance(value, list):
        for item in value:
            yield from strings(item)


class RerankExampleTests(unittest.TestCase):
    def test_inputs_are_exact_primary_source_excerpts(self) -> None:
        cases, sources = run.load_and_verify_inputs()
        self.assertEqual(len(cases["cases"]), 4)
        self.assertEqual(len(sources["sources"]), 4)

    def test_recorded_audit_envelopes_match_the_runner(self) -> None:
        cases, _ = run.load_and_verify_inputs()
        for case_id, case in cases["cases"].items():
            name = run.ARTIFACT_NAMES[case_id]
            recorded = run.read_json(ROOT / "verified-run" / "requests" / f"{name}.json")
            self.assertEqual(recorded, run.build_audit_envelope(case_id, case))

    def test_recorded_responses_pass_fail_closed_checks(self) -> None:
        cases, _ = run.load_and_verify_inputs()
        evaluation = run.read_json(ROOT / "verified-run" / "evaluation.json")
        checks = {check["case_id"]: check for check in evaluation["checks"]}
        for case_id, case in cases["cases"].items():
            name = run.ARTIFACT_NAMES[case_id]
            response = run.read_json(ROOT / "verified-run" / "raw" / f"{name}.json")
            observed = run.validate_response(case_id, case, response)
            self.assertTrue(observed["passed"])
            self.assertEqual(
                checks[case_id]["expected_top_candidate_id"],
                observed["expected_top_candidate_id"],
            )
            self.assertEqual(
                checks[case_id]["observed_top_candidate_id"],
                observed["observed_top_candidate_id"],
            )
            self.assertEqual(
                checks[case_id]["candidate_count"],
                observed["candidate_count"],
            )

    def test_boolean_score_fails_closed(self) -> None:
        cases, _ = run.load_and_verify_inputs()
        case_id = "scotus_two_contracts"
        case = cases["cases"][case_id]
        response = run.read_json(ROOT / "verified-run" / "raw" / "supreme-court-arbitration.json")
        response["scores"][0]["score"] = True

        with self.assertRaisesRegex(ValueError, "Invalid score"):
            run.validate_response(case_id, case, response)

    def test_manifest_rejects_rewritten_excerpt_and_declared_hash(self) -> None:
        cases = run.read_json(run.CASES_PATH)
        sources = run.read_json(run.SOURCES_PATH)
        candidate = cases["cases"]["scotus_two_contracts"]["candidates"][0]
        candidate["text"] += " tampered"
        candidate["sha256"] = run.sha256_bytes(candidate["text"].encode("utf-8"))

        with (
            mock.patch.object(run, "read_json", side_effect=[cases, sources]),
            self.assertRaisesRegex(ValueError, "Canonical excerpt changed"),
        ):
            run.load_and_verify_inputs()

    def test_manifest_pins_every_artifact(self) -> None:
        manifest = run.read_json(ROOT / "verified-run" / "manifest.json")
        self.assertEqual(manifest["inputs"]["cases_sha256"], run.sha256_file(run.CASES_PATH))
        self.assertEqual(manifest["inputs"]["sources_sha256"], run.sha256_file(run.SOURCES_PATH))
        self.assertEqual(set(manifest["artifacts"]), set(run.ARTIFACT_NAMES))
        for artifact in manifest["artifacts"].values():
            self.assertEqual(
                artifact["request_sha256"],
                run.sha256_file(ROOT / artifact["request"]),
            )
            self.assertEqual(
                artifact["raw_response_sha256"],
                run.sha256_file(ROOT / artifact["raw_response"]),
            )
        evaluation = manifest["evaluation"]
        self.assertEqual(evaluation["sha256"], run.sha256_file(ROOT / evaluation["file"]))

    def test_metadata_has_no_temporary_filesystem_paths(self) -> None:
        forbidden = ("/Users/", "/root/", "/tmp/", "reference-batch", "/v4/")
        for directory in (ROOT / "data", ROOT / "verified-run"):
            for path in directory.rglob("*.json"):
                value = json.loads(path.read_text(encoding="utf-8"))
                for text in strings(value):
                    self.assertFalse(
                        any(marker in text for marker in forbidden),
                        f"{path}: {text}",
                    )


if __name__ == "__main__":
    unittest.main()
