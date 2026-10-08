"""Offline checks for the scoring and publication contracts."""

from __future__ import annotations

import hashlib
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from record import OneAttempt, public_usage
from score import LABELS, MODELS, ROOT, evaluate, load_frame, tag_set


class ScoringTests(unittest.TestCase):
    def test_frozen_frame(self) -> None:
        cases, variants = load_frame(ROOT / "inputs/cases.jsonl", ROOT / "inputs/acceptable-tag-representations.json")
        self.assertEqual(len(cases), 24)
        self.assertEqual(len(variants), 7)
        self.assertEqual(sum(bool(case["tags"]) for case in cases), 17)

    def test_failed_empty_and_missing_calls_score_zero(self) -> None:
        cases = [
            {"id": "empty", "domain": "test", "text": "No names.", "tags": []},
            {"id": "positive", "domain": "test", "text": "Ada spoke.", "tags": [{"text": "Ada", "type": "person"}]},
        ]
        report = evaluate(cases, {}, [{"id": "empty", "model": "comparison", "status": "failed"}])
        self.assertEqual(report["macro_f1"], 0)
        self.assertEqual(report["successful_calls"], 0)
        self.assertEqual(report["cases"][1]["fn"], 1)
        self.assertFalse(report["useful_fit_gate"])

    def test_fixed_complete_variant_not_individual_tag_union(self) -> None:
        text = "RESTON, Va."
        digest = hashlib.sha256(text.encode()).hexdigest()
        strict = [{"text": "RESTON", "type": "location"}, {"text": "Va.", "type": "location"}]
        combined = {("RESTON, Va.", "location")}
        cases = [{"id": "city", "domain": "test", "text": text, "tags": strict, "text_sha256": digest}]
        rows = [
            {
                "id": "city",
                "model": "comparison",
                "status": "ok",
                "text_sha256": digest,
                "entities": [{"text": text, "type": "location"}],
            }
        ]
        report = evaluate(cases, {"city": [tag_set(text, strict), combined]}, rows)
        self.assertEqual(report["macro_f1"], 1)
        self.assertEqual(report["strict_macro_f1"], 0)
        rows[0]["entities"].extend(strict)
        mixed = evaluate(cases, {"city": [tag_set(text, strict), combined]}, rows)
        self.assertLess(mixed["macro_f1"], 1)

    def test_unicode_offsets_and_distinct_tags(self) -> None:
        entity = {"text": "Ada", "label": "person", "start": 2, "end": 5, "score": 0.9}
        self.assertEqual(tag_set("🐍 Ada Ada", [entity, entity], spans=True), {("Ada", "person")})
        with self.assertRaises(ValueError):
            tag_set("🐍 Ada Ada", [{**entity, "start": 3}], spans=True)

    def test_invented_comparison_name_is_a_false_positive(self) -> None:
        text = "Ada spoke."
        cases = [
            {
                "id": "name",
                "domain": "test",
                "text": text,
                "tags": [{"text": "Ada", "type": "person"}],
                "text_sha256": hashlib.sha256(text.encode()).hexdigest(),
            }
        ]
        rows = [
            {
                "id": "name",
                "model": "comparison",
                "status": "ok",
                "text_sha256": cases[0]["text_sha256"],
                "entities": [{"text": "Ada Lovelace", "type": "person"}],
            }
        ]
        report = evaluate(cases, {}, rows)
        self.assertEqual(report["cases"][0]["fp"], 1)
        self.assertEqual(report["cases"][0]["fn"], 1)
        self.assertEqual(report["macro_f1"], 0)

    def test_retry_fence_and_usage_allowlist(self) -> None:
        fence = OneAttempt()
        request = SimpleNamespace(method="POST")
        fence(request)
        with self.assertRaises(RuntimeError):
            fence(request)
        self.assertEqual(fence.posts, 1)
        self.assertEqual(
            public_usage({"request": {"id": "private", "usage": {"input_tokens": 9, "secret": "private"}}}),
            {"input_tokens": 9},
        )
        self.assertEqual(LABELS, ("person", "organization", "location"))
        self.assertEqual(len(MODELS), 2)


if __name__ == "__main__":
    unittest.main()
