from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import run


def strings(value: Any):
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for item in value.values():
            yield from strings(item)
    elif isinstance(value, list):
        for item in value:
            yield from strings(item)


class NamedEntityExampleTests(unittest.TestCase):
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

    def test_all_recorded_spans_match_the_source_offsets(self) -> None:
        cases, _ = run.load_and_verify_inputs()
        recorded_evaluation = run.read_json(ROOT / "verified-run" / "evaluation.json")
        checks = {check["case_id"]: check for check in recorded_evaluation["checks"]}
        count = 0
        for case_id, case in cases["cases"].items():
            name = run.ARTIFACT_NAMES[case_id]
            response = run.read_json(ROOT / "verified-run" / "raw" / f"{name}.json")
            evaluation = run.validate_response(case_id, case, response)
            self.assertTrue(evaluation["passed"])
            self.assertEqual(checks[case_id]["entity_count"], evaluation["entity_count"])
            self.assertEqual(
                checks[case_id]["returned_labels"],
                evaluation["returned_labels"],
            )
            self.assertEqual(
                checks[case_id]["required_anchor_count"],
                evaluation["required_anchor_count"],
            )
            self.assertEqual(
                checks[case_id]["matched_anchor_count"],
                evaluation["matched_anchor_count"],
            )
            self.assertEqual(
                checks[case_id]["anchor_checks"],
                evaluation["anchor_checks"],
            )
            count += evaluation["entity_count"]
        self.assertEqual(count, 53)

    def test_empty_response_fails_required_anchor_check(self) -> None:
        cases, _ = run.load_and_verify_inputs()
        case_id = "scotus_two_contracts"
        response = run.read_json(ROOT / "verified-run" / "raw" / "supreme-court-caption.json")
        response["entities"] = []
        with self.assertRaisesRegex(ValueError, "missing required anchors"):
            run.validate_response(case_id, cases["cases"][case_id], response)

    def test_cms_false_positive_is_preserved_but_not_required(self) -> None:
        cases, _ = run.load_and_verify_inputs()
        response = run.read_json(ROOT / "verified-run" / "raw" / "cms-orthosis-documentation.json")
        false_positive = {
            "text": "proof of delivery",
            "label": "missing documentation",
            "start": 480,
            "end": 497,
        }
        self.assertTrue(
            any(
                all(entity[field] == value for field, value in false_positive.items())
                for entity in response["entities"]
            )
        )
        self.assertNotIn(
            tuple(false_positive[field] for field in run.ANCHOR_FIELDS),
            {run.anchor_key(anchor) for anchor in cases["cases"]["cms_lower_limb_orthosis"]["required_anchors"]},
        )

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
