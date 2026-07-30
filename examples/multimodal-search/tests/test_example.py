from __future__ import annotations

import copy
import importlib.util
import json
import sys
import unittest
from pathlib import Path
from typing import Any
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
RUN_SPEC = importlib.util.spec_from_file_location("multimodal_search_run", ROOT / "run.py")
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


class MultimodalSearchExampleTests(unittest.TestCase):
    def test_images_match_the_licensed_source_manifest(self) -> None:
        sources = run.load_and_verify_sources()
        self.assertEqual(len(sources["images"]), 6)
        self.assertTrue(
            all(
                image["license"] and image["license_url"] and image["source"] and image["creator"]
                for image in sources["images"]
            )
        )

    def test_recorded_audit_envelopes_match_the_runner(self) -> None:
        sources = run.load_and_verify_sources()
        recorded_query = run.read_json(ROOT / "verified-run" / "requests" / "text-query.json")
        recorded_images = run.read_json(ROOT / "verified-run" / "requests" / "image-candidates.json")
        self.assertEqual(recorded_query, run.build_query_audit_envelope(sources))
        self.assertEqual(recorded_images, run.build_image_audit_envelope(sources))

    def test_recorded_ranking_is_recomputed_from_full_vectors(self) -> None:
        sources = run.load_and_verify_sources()
        query = run.read_json(ROOT / "verified-run" / "raw" / "text-query-embedding.json")
        images = run.read_json(ROOT / "verified-run" / "raw" / "image-candidate-embeddings.json")
        observed = run.evaluate(sources, query, images)
        recorded = run.read_json(ROOT / "verified-run" / "evaluation.json")
        self.assertEqual(observed["query"], recorded["query"])
        self.assertEqual(observed["metric"], recorded["metric"])
        self.assertEqual(observed["image_count"], recorded["image_count"])
        for left, right in zip(
            observed["sorted_matches"],
            recorded["sorted_matches"],
            strict=True,
        ):
            self.assertEqual(left["rank"], right["rank"])
            self.assertEqual(left["file"], right["file"])
            self.assertEqual(left["sha256"], right["sha256"])
            self.assertAlmostEqual(left["score"], right["score"], places=15)

    def test_source_manifest_fails_closed_on_tampering(self) -> None:
        sources = run.load_and_verify_sources()

        synthetic = copy.deepcopy(sources)
        synthetic["synthetic_or_generated_images"] = True
        with (
            patch.object(run, "read_json", return_value=synthetic),
            self.assertRaisesRegex(ValueError, "reject synthetic images"),
        ):
            run.load_and_verify_sources()

        incomplete = copy.deepcopy(sources)
        incomplete["images"].pop()
        with (
            patch.object(run, "read_json", return_value=incomplete),
            self.assertRaisesRegex(ValueError, "exactly six"),
        ):
            run.load_and_verify_sources()

        wrong_checksum = copy.deepcopy(sources)
        wrong_checksum["images"][0]["sha256"] = "0" * 64
        with (
            patch.object(run, "read_json", return_value=wrong_checksum),
            self.assertRaisesRegex(ValueError, "checksum changed"),
        ):
            run.load_and_verify_sources()

    def test_dense_vector_rejects_non_finite_values(self) -> None:
        vector = [0.0] * run.EXPECTED_DIMENSIONS
        vector[-1] = float("nan")
        with self.assertRaisesRegex(ValueError, "non-finite"):
            run.dense_vector({"dense": vector})

    def test_evaluation_rejects_an_unexpected_top_match(self) -> None:
        sources = run.load_and_verify_sources()
        query_vector = [1.0, *([0.0] * (run.EXPECTED_DIMENSIONS - 1))]
        other_vector = [0.0, 1.0, *([0.0] * (run.EXPECTED_DIMENSIONS - 2))]
        image_responses = [
            {"dense": (other_vector if Path(image["file"]).name == run.EXPECTED_TOP_MATCH else query_vector)}
            for image in sources["images"]
        ]

        with self.assertRaisesRegex(ValueError, "Expected red-leather-handbag.png"):
            run.evaluate(sources, {"dense": query_vector}, image_responses)

    def test_manifest_pins_inputs_and_artifacts(self) -> None:
        manifest = run.read_json(ROOT / "verified-run" / "manifest.json")
        self.assertEqual(
            manifest["inputs"]["sources_sha256"],
            run.sha256_file(run.SOURCES_PATH),
        )
        expected_images = {f"data/{image['file']}" for image in run.load_and_verify_sources()["images"]}
        self.assertEqual(set(manifest["inputs"]["image_files"]), expected_images)
        for relative, digest in manifest["inputs"]["image_files"].items():
            self.assertEqual(digest, run.sha256_file(ROOT / relative))
        expected_artifacts = {
            "verified-run/evaluation.json",
            "verified-run/raw/image-candidate-embeddings.json",
            "verified-run/raw/text-query-embedding.json",
            "verified-run/requests/image-candidates.json",
            "verified-run/requests/text-query.json",
        }
        self.assertEqual(set(manifest["artifacts"]), expected_artifacts)
        for relative, digest in manifest["artifacts"].items():
            self.assertEqual(digest, run.sha256_file(ROOT / relative))

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
