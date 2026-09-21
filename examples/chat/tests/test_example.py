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

import prompt
import score


@contextmanager
def patched_json(path: Path, value: Any):
    """Make prompt.read_json return `value` for one path, unchanged elsewhere."""
    original = prompt.read_json

    def replacement(target: Path) -> Any:
        return value if target == path else original(target)

    prompt.read_json = replacement
    try:
        yield
    finally:
        prompt.read_json = original


class ChatExampleTests(unittest.TestCase):
    def test_twelve_pinned_reports_match_their_digests(self) -> None:
        cases = prompt.load_cases()
        self.assertEqual(len(cases["cases"]), 12)
        for case in cases["cases"]:
            self.assertTrue((ROOT / case["source_file"]).exists(), case["slug"])

    def test_every_recorded_answer_passes_all_four_checks(self) -> None:
        summary = score.score()
        self.assertEqual(summary["reports_scored"], 12)
        self.assertEqual(summary["passing_all_checks"], 12)
        for name in score.CHECK_NAMES:
            self.assertEqual(summary["check_passes"][name], 12, name)

    def test_recorded_calls_name_the_endpoint_model_and_revision(self) -> None:
        calls = json.loads((ROOT / "calls.json").read_text(encoding="utf-8"))
        cases = json.loads((ROOT / "data" / "cases.json").read_text(encoding="utf-8"))
        self.assertEqual(calls["endpoint"], "https://api.superlinked.com")
        self.assertEqual(calls["path"], prompt.CHAT_COMPLETIONS_PATH)
        self.assertEqual(calls["model"], cases["model"])
        revisions = set()
        for entry in calls["calls"]:
            self.assertEqual(entry["request"]["body"]["model"], calls["model"])
            self.assertTrue(entry["model_revision"], entry["slug"])
            revisions.add(entry["model_revision"])
        self.assertEqual(revisions, {calls["recording"]["model_revision"]})

    def test_a_missing_source_file_fails_rather_than_skipping(self) -> None:
        cases = json.loads((ROOT / "data" / "cases.json").read_text(encoding="utf-8"))
        cases["cases"][0]["source_file"] = "data/sources/not-committed.wiki"
        with patched_json(prompt.CASES_PATH, cases), self.assertRaisesRegex(prompt.InputError, "missing"):
            prompt.load_cases()

    def test_a_missing_recorded_call_fails_rather_than_skipping(self) -> None:
        calls = json.loads((ROOT / "calls.json").read_text(encoding="utf-8"))
        dropped = calls["calls"].pop()["slug"]
        with patched_json(prompt.CALLS_PATH, calls), self.assertRaisesRegex(prompt.InputError, dropped):
            score.score()

    def test_an_edited_response_fails_its_digest(self) -> None:
        calls = json.loads((ROOT / "calls.json").read_text(encoding="utf-8"))
        message = calls["calls"][0]["response"]["choices"][0]["message"]
        message["content"] = message["content"].replace("Resolved", "Ongoing")
        with patched_json(prompt.CALLS_PATH, calls), self.assertRaisesRegex(prompt.InputError, "response_sha256"):
            score.score()

    def test_a_rewritten_source_no_longer_rebuilds_the_recorded_request(self) -> None:
        """A matching digest is not the only binding: the text must build the call."""
        cases = prompt.load_cases()
        case = next(entry for entry in cases["cases"] if entry["slug"] == "sessionstore")
        original = prompt.wikitext(case)
        edited = original.replace("disk exhaustion", "disk starvation", 1)
        self.assertNotEqual(edited, original, "the tamper string is no longer in the pinned source")

        calls = json.loads((ROOT / "calls.json").read_text(encoding="utf-8"))
        recorded = next(entry for entry in calls["calls"] if entry["slug"] == case["slug"])
        self.assertEqual(prompt.request_body(cases, original, case), recorded["request"]["body"])
        self.assertNotEqual(
            prompt.request_body(cases, edited, case)["messages"][1]["content"],
            recorded["request"]["body"]["messages"][1]["content"],
        )

    def test_a_leaked_hostname_would_fail_its_case(self) -> None:
        """The no-internal-identifiers check can fail, so 12 of 12 means something."""
        cases = prompt.load_cases()
        case = next(entry for entry in cases["cases"] if entry["slug"] == "gerrit-data-corruption")
        source = prompt.wikitext(case)
        answer = (
            "Status: Resolved\n"
            "Impact: Gerrit was unavailable for five hours.\n"
            "Window: 2025-04-30 15:42 to 2025-04-30 21:19 UTC\n"
            "Cause: A failed switchover on gerrit1003 left two hosts acting as primary."
        )
        result = score.evaluate(case, source, answer)
        self.assertFalse(result["checks"]["no_internal_identifiers"])
        self.assertIn("gerrit1003", [leak["text"] for leak in result["details"]["leaks"]])
        self.assertTrue(result["checks"]["format"])
        self.assertTrue(result["checks"]["window"])

    def test_no_api_key_or_local_path_leaked_into_the_recording(self) -> None:
        forbidden = ("sk-sie-", "/Users/", "/root/", "Authorization", "Bearer ")
        for path in (ROOT / "calls.json", ROOT / "data" / "cases.json"):
            text = path.read_text(encoding="utf-8")
            for marker in forbidden:
                self.assertNotIn(marker, text, f"{path.name}: {marker}")


if __name__ == "__main__":
    unittest.main()
