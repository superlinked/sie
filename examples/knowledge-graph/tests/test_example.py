"""Offline tests. They never call a server and need nothing installed.

python3 -m unittest discover -s tests -v
"""

from __future__ import annotations

import json
import sys
import unittest
from contextlib import contextmanager
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import graph
import score


@contextmanager
def patched_json(path: Path, value: Any):
    """Make graph.read_json return `value` for one path, unchanged elsewhere."""
    original = graph.read_json

    def replacement(target: Path) -> Any:
        return value if target == path else original(target)

    graph.read_json = replacement
    try:
        yield
    finally:
        graph.read_json = original


def load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


class KnowledgeGraphExampleTests(unittest.TestCase):
    def test_ten_paragraphs_match_their_digests(self) -> None:
        doc = graph.load_candidates()
        self.assertEqual(len(doc["candidates"]), 10)
        self.assertEqual(len(graph.shown(doc)), 5)

    def test_published_counts(self) -> None:
        summary = score.score()
        self.assertEqual(summary["candidates_recorded"], 10)
        self.assertEqual(summary["candidates_shown"], 5)
        self.assertEqual(summary["hero_edges"], 5)
        self.assertEqual(summary["proof_edges"], 11)
        self.assertEqual(summary["edges_drawn"], 16)
        self.assertEqual(summary["flagged_edges"], 3)

    def test_every_paragraph_has_both_calls_at_the_documented_endpoint(self) -> None:
        calls = load(ROOT / "calls.json")
        candidates = load(ROOT / "data" / "candidates.json")
        self.assertEqual(calls["endpoint"], "https://api.superlinked.com")
        self.assertEqual(calls["path"], graph.extract_path(candidates["model"]))
        self.assertEqual(len(calls["calls"]), 20)
        revisions = set()
        for entry in calls["calls"]:
            self.assertEqual(entry["request"]["model"], candidates["model"])
            revisions.add(entry["model_revision"])
        self.assertEqual(revisions, {calls["recording"]["model_hf_revision"]})

    def test_a_missing_call_fails_rather_than_skipping(self) -> None:
        calls = load(ROOT / "calls.json")
        dropped = next(entry for entry in calls["calls"] if entry["kind"] == "relations")
        calls["calls"] = [entry for entry in calls["calls"] if entry is not dropped]
        with patched_json(graph.CALLS_PATH, calls), self.assertRaisesRegex(graph.InputError, "no recorded relations"):
            score.score()

    def test_an_edited_response_fails_its_digest(self) -> None:
        calls = load(ROOT / "calls.json")
        entry = next(entry for entry in calls["calls"] if entry["kind"] == "relations")
        entry["response"]["item"]["relations"].append(
            {"head": "Invented", "relation": "acquired", "tail": "Fabricated", "score": 1.0}
        )
        with patched_json(graph.CALLS_PATH, calls), self.assertRaisesRegex(graph.InputError, "response_sha256"):
            score.score()

    def test_an_edge_the_model_never_returned_cannot_be_flagged(self) -> None:
        review = load(ROOT / "data" / "review.json")
        review["flagged"][0]["tail"] = "Nevada"
        with patched_json(graph.REVIEW_PATH, review), self.assertRaisesRegex(graph.InputError, "returned 0 times"):
            score.score()

    def test_a_flag_on_a_hidden_paragraph_is_refused(self) -> None:
        """The flagged count is 3 of the 11 proof edges, so it cannot borrow from the rest."""
        review = load(ROOT / "data" / "review.json")
        review["flagged"][0]["candidate"] = "nvidia-supply-chain"
        with patched_json(graph.REVIEW_PATH, review), self.assertRaisesRegex(graph.InputError, "not a proof"):
            score.score()

    def test_the_relations_call_carries_the_entities_call_output(self) -> None:
        """The two calls are chained, and the recording proves it rather than asserting it."""
        calls = load(ROOT / "calls.json")
        by_key = {(entry["candidate"], entry["kind"]): entry for entry in calls["calls"]}
        for candidate in graph.load_candidates()["candidates"]:
            entities = by_key[(candidate["id"], "entities")]["response"]["item"]["entities"]
            metadata = by_key[(candidate["id"], "relations")]["request"]["body"]["items"][0]["metadata"]
            self.assertEqual(metadata["entities"], entities, candidate["id"])

    def test_rewritten_text_no_longer_rebuilds_the_recorded_request(self) -> None:
        doc = graph.load_candidates()
        candidate = dict(next(c for c in doc["candidates"] if c["id"] == "flex-credit-facility"))
        calls = load(ROOT / "calls.json")
        recorded = next(
            entry for entry in calls["calls"] if entry["candidate"] == candidate["id"] and entry["kind"] == "entities"
        )
        self.assertEqual(graph.entities_body(candidate), recorded["request"]["body"])
        candidate["text"] = candidate["text"].replace("Citibank", "Citibank, National Association", 1)
        self.assertNotEqual(graph.entities_body(candidate), recorded["request"]["body"])

    def test_no_api_key_or_local_path_leaked_into_the_recording(self) -> None:
        forbidden = ("sk-sie-", "/Users/", "/root/", "Authorization", "Bearer ")
        for path in (ROOT / "calls.json", ROOT / "data" / "candidates.json", ROOT / "data" / "review.json"):
            text = path.read_text(encoding="utf-8")
            for marker in forbidden:
                self.assertNotIn(marker, text, f"{path.name}: {marker}")


if __name__ == "__main__":
    unittest.main()
