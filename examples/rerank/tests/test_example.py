"""Integrity tests for the public recording; fetch.py must run first."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import shutil
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = Path(os.environ.get("RERANK_EVIDENCE_DIR", str(ROOT / "data")))
spec = importlib.util.spec_from_file_location("rerank_score", ROOT / "score.py")
assert spec and spec.loader
scorer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(scorer)
fetch_spec = importlib.util.spec_from_file_location("rerank_fetch", ROOT / "fetch.py")
assert fetch_spec and fetch_spec.loader
fetcher = importlib.util.module_from_spec(fetch_spec)
fetch_spec.loader.exec_module(fetcher)


class RecordedEvidenceTests(unittest.TestCase):
    def test_published_figures_and_completeness(self):
        result = scorer.score(DATA)
        self.assertEqual(result["models"]["sie-qwen3-reranker-4b"]["bothRulesAllRepetitions"], 455)
        self.assertEqual(result["models"]["gpt-6-luna"]["proposal"], 491)
        self.assertEqual(result["models"]["claude-haiku-4-5"]["partialRankings"], 668)

    def tamper(self, name, change, expected):
        with tempfile.TemporaryDirectory() as temporary:
            target = Path(temporary) / "data"
            shutil.copytree(DATA, target)
            path = target / name
            change(path)
            manifest = scorer.load(target / "manifest.json")
            manifest["files_sha256"][name] = hashlib.sha256(path.read_bytes()).hexdigest()
            (target / "manifest.json").write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, expected):
                scorer.score(target)

    def test_missing_call_keeps_fixed_denominator(self):
        self.tamper(
            "calls/sie-qwen3-reranker-4b.jsonl",
            lambda p: p.write_text("\n".join(p.read_text().splitlines()[1:]) + "\n"),
            "Missing, duplicate",
        )

    def test_duplicate_call_is_not_an_extra_success(self):
        self.tamper(
            "calls/sie-qwen3-reranker-4b.jsonl",
            lambda p: p.write_text(p.read_text() + p.read_text().splitlines()[0] + "\n"),
            "Missing, duplicate",
        )

    def test_declared_winner_must_match_recorded_order(self):
        def change(path):
            rows = scorer.rows(path)
            rows[0]["top1"] = rows[0]["ranked_ids"][1]
            path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")

        self.tamper("calls/sie-qwen3-reranker-4b.jsonl", change, "Winner differs")

    def test_repair_cannot_replace_an_original_success(self):
        def change(path):
            rows = scorer.rows(path)
            rows[-1]["case"] = rows[0]["case"]
            rows[-1]["arm"] = rows[0]["arm"]
            path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")

        self.tamper("history/transport-repair.jsonl", change, "Retries must only")

    def test_fetch_refuses_unowned_destination_and_existing_backup(self):
        with tempfile.TemporaryDirectory() as temporary:
            target = Path(temporary) / "data"
            target.mkdir()
            self.assertIn("no .sie-evidence", fetcher.refuse_reason(target))
            (target / ".sie-evidence").write_bytes(fetcher.marker_bytes())
            self.assertIsNone(fetcher.refuse_reason(target))
            fetcher.backup_path(target).mkdir()
            self.assertIn("already exists", fetcher.refuse_reason(target))

    def test_unverified_bytes_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            target = Path(temporary) / "data"
            shutil.copytree(DATA, target)
            (target / "analysis.json").write_text("{}")
            with self.assertRaisesRegex(ValueError, "Digest differs"):
                scorer.score(target)


if __name__ == "__main__":
    unittest.main()
