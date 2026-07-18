from __future__ import annotations

import hashlib
import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
REPO_ROOT = ROOT.parents[1]
PACK = json.loads((ROOT / "course-pack.json").read_text())


def _load_script(name: str) -> Any:
    path = ROOT / "scripts" / name
    spec = importlib.util.spec_from_file_location(path.stem, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class CoursePackContractTest(unittest.TestCase):
    def test_three_unique_selected_projects(self) -> None:
        projects = PACK["projects"]
        self.assertEqual(len(projects), 3)
        self.assertEqual(len({project["id"] for project in projects}), 3)
        self.assertEqual(
            {project["id"] for project in projects},
            {"contract-review", "vision-document-rag", "retrieval-ablation"},
        )

    def test_source_paths_and_bounded_samples_exist(self) -> None:
        for project in PACK["projects"]:
            self.assertTrue((ROOT / project["source"]).resolve().is_dir())
            sample = project["sample"]
            self.assertLessEqual(sample["max_run_items"], sample["max_corpus_items"])
            self.assertGreater(sample["max_run_items"], 0)

    def test_unmeasured_catalog_and_budget_values_are_null(self) -> None:
        expected_roles = {
            "contract-review": {
                "triage",
                "orchestrator",
                "vision",
                "reasoning",
                "sql",
                "guard",
                "ocr",
                "embed",
                "rerank",
                "entities",
            },
            "vision-document-rag": {"retriever", "reranker", "answer", "sglang_ocr"},
            "retrieval-ablation": {"embed", "rerank"},
        }
        for project in PACK["projects"]:
            self.assertEqual(project["model_gate"]["status"], "catalog_freeze_required")
            self.assertEqual(
                {role["role"] for role in project["model_gate"]["roles"]},
                expected_roles[project["id"]],
            )
            for role in project["model_gate"]["roles"]:
                self.assertIsNone(role["model_id"])
                self.assertIsNone(role["revision"])
            budget = project["budget"]
            self.assertEqual(budget["status"], "live_measurement_required")
            self.assertIsNone(budget["cold_run_credits"])
            self.assertIsNone(budget["warm_run_credits"])
            self.assertIsNone(budget["student_allocation_credits"])

    def test_each_project_declares_change_measure_and_output_contract(self) -> None:
        for project in PACK["projects"]:
            self.assertGreaterEqual(len(project["what_to_change"]), 2)
            self.assertIn("request_ids", project["what_to_measure"])
            self.assertIn("settled_credits", project["what_to_measure"])
            self.assertTrue(project["expected_output"]["required_top_level"])

    def test_pinned_fixture_hashes(self) -> None:
        vision = next(project for project in PACK["projects"] if project["id"] == "vision-document-rag")
        for fixture in vision["sample"]["fixtures"]:
            path = (ROOT / fixture["path"]).resolve()
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), fixture["sha256"])
        contract = next(project for project in PACK["projects"] if project["id"] == "contract-review")
        generator = (ROOT / contract["source"] / "contract_review_agent/data/make_sample.py").resolve()
        self.assertEqual(hashlib.sha256(generator.read_bytes()).hexdigest(), contract["sample"]["source_sha256"])

    def test_retrieval_fixture_qrels_are_valid_and_bounded(self) -> None:
        fixture = json.loads((ROOT / "fixtures/retrieval-course.json").read_text())
        document_ids = {document["id"] for document in fixture["documents"]}
        query_ids = {query["id"] for query in fixture["queries"]}
        self.assertEqual(len(document_ids), 12)
        self.assertEqual(len(query_ids), 6)
        for query in fixture["queries"]:
            self.assertTrue(query["relevant"])
            self.assertLessEqual(set(query["relevant"]), document_ids)
            self.assertTrue(all(score > 0 for score in query["relevant"].values()))

    def test_offline_ablation_output_contract(self) -> None:
        module = _load_script("course_ablation.py")
        output = module.offline_contract(module.load_fixture())
        self.assertEqual(output["schema_version"], "abi-course-ablation/v1")
        self.assertEqual(output["mode"], "offline-contract")
        self.assertEqual(output["dataset"], {"documents": 12, "queries": 6})
        self.assertEqual(output["measurement_gate"], "offline_only")
        condition = output["conditions"][0]
        for field in (
            "name",
            "models",
            "metrics",
            "latency_ms",
            "request_count",
            "request_ids",
            "usage",
            "settled_credits",
        ):
            self.assertIn(field, condition)
        self.assertEqual(condition["request_count"], 0)
        self.assertEqual(condition["models"], [])
        self.assertEqual(condition["request_ids"], [])
        self.assertIsNone(condition["usage"])
        self.assertIsNone(condition["settled_credits"])
        self.assertGreaterEqual(condition["metrics"]["ndcg_at_3"], 0)
        self.assertLessEqual(condition["metrics"]["ndcg_at_3"], 1)

    def test_offline_cli_emits_only_json(self) -> None:
        completed = subprocess.run(
            [sys.executable, str(ROOT / "scripts/course_ablation.py")],
            check=True,
            capture_output=True,
            text=True,
        )
        output = json.loads(completed.stdout)
        self.assertEqual(output["mode"], "offline-contract")
        self.assertEqual(completed.stderr, "")

    def test_vision_preparation_is_bounded_and_deterministic(self) -> None:
        module = _load_script("prepare_vision_sample.py")
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            first = module.prepare(output)
            second = module.prepare(output)
            self.assertEqual(first, second)
            self.assertEqual(len(first), 3)
            self.assertEqual({row["client"] for row in first}, {"course-docs"})
            self.assertEqual(
                {row["image_path"] for row in first},
                {"pages/course/receipt.png", "pages/course/invoice.png", "pages/course/slide.png"},
            )
            self.assertEqual(json.loads((output / "pages_manifest.json").read_text()), first)

    def test_pack_does_not_contain_key_shaped_literals(self) -> None:
        forbidden = ("SL-" + r"[A-Za-z0-9]{12,}", "api[_-]?key\\s*[:=]\\s*['\"][^'\"]+")
        for path in ROOT.rglob("*"):
            if path.is_file() and path.suffix in {".py", ".md", ".json", ".toml"}:
                text = path.read_text()
                for pattern in forbidden:
                    self.assertNotRegex(text, pattern, msg=f"credential-shaped literal in {path}")


if __name__ == "__main__":
    unittest.main()
