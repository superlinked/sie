"""Tamper tests for the offline checks, carried over from the committed run.

    python3 fetch.py
    python3 -m unittest discover -s tests -v

Standard library only, and no server. These were written when the recorded run
lived in this repository; they now read the fetched dataset instead. Nothing
here is new: each one still asserts that a specific forgery fails closed.

A missing `data/` directory fails these tests rather than skipping them. A
check that cannot run has not passed.
"""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import sys
import unittest
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
REQUIRED = ("inputs/cases.json", "inputs/sources.json", "calls.json", "manifest.json")


def _module(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


run = _module("run")
score = _module("score")


class RerankExampleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        missing = [name for name in REQUIRED if not (DATA / name).exists()]
        if missing:
            raise AssertionError(f"fetch the evidence first (python3 fetch.py); missing {', '.join(missing)}")
        cls.cases = json.loads((DATA / "inputs/cases.json").read_text(encoding="utf-8"))
        cls.sources = json.loads((DATA / "inputs/sources.json").read_text(encoding="utf-8"))
        cls.calls = json.loads((DATA / "calls.json").read_text(encoding="utf-8"))
        cls.manifest = json.loads((DATA / "manifest.json").read_text(encoding="utf-8"))

    def test_manifest_pins_every_fetched_file(self) -> None:
        """The digest chain fetch.py walked, checked again after the download.

        fetch.py pins manifest.json by the digest in its own source and then
        pins every other file by the digest inside that manifest, so the bytes
        are authenticated before anything parses them. This re-walks the second
        half against what actually landed on disk.
        """
        pinned = self.manifest["files_sha256"]
        self.assertEqual(pinned["calls.json"], self.manifest["calls_sha256"])
        on_disk = {
            str(path.relative_to(DATA)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(DATA.rglob("*"))
            if path.is_file() and path.name not in {"manifest.json", ".sie-evidence"}
        }
        self.assertEqual(on_disk, pinned)

    def test_inputs_are_exact_primary_source_excerpts(self) -> None:
        score.verify_inputs(self.cases, self.sources)
        self.assertEqual(len(self.cases["cases"]), 4)
        self.assertEqual(len(self.sources["sources"]), 4)

    def test_recorded_envelopes_match_the_runner(self) -> None:
        self.assertEqual(run.check(DATA), 0)

    def test_boolean_score_fails_closed(self) -> None:
        case_id = "scotus_two_contracts"
        case = self.cases["cases"][case_id]
        body = copy.deepcopy(self._body(case_id))
        body["scores"][0]["score"] = True
        with self.assertRaisesRegex(SystemExit, "non-numeric score"):
            score.ranked(case_id, case, body)

    def test_a_candidate_scored_twice_fails_closed(self) -> None:
        """A count is not a set.

        Replacing one candidate's row with a duplicate of another keeps the row
        count and the 0..n-1 ranks intact, so only comparing the identities
        catches it.
        """
        case_id = "ntsb_detector_alert"
        case = self.cases["cases"][case_id]
        body = copy.deepcopy(self._body(case_id))
        body["scores"][0]["item_id"] = body["scores"][1]["item_id"]
        with self.assertRaisesRegex(SystemExit, "candidates"):
            score.ranked(case_id, case, body)

    def test_rewritten_excerpt_fails_even_with_its_declared_hash(self) -> None:
        """The forgery a single declared digest cannot catch.

        Rewrite the text and the digest recorded beside it together and that
        pair agrees with itself. The canonical digest in sources.json is what
        refuses it, which is why both files travel together.
        """
        cases = copy.deepcopy(self.cases)
        candidate = cases["cases"]["scotus_two_contracts"]["candidates"][0]
        candidate["text"] += " tampered"
        candidate["sha256"] = score.sha256_text(candidate["text"])
        with self.assertRaisesRegex(SystemExit, "canonical one in sources.json"):
            score.verify_inputs(cases, self.sources)

    def test_a_failed_run_is_never_scored(self) -> None:
        payload = copy.deepcopy(self.calls)
        payload["complete"] = False
        payload["failed_calls"] = 1
        with self.assertRaisesRegex(SystemExit, "not a complete run"):
            score.scored_calls(payload)

    def test_metadata_has_no_temporary_filesystem_paths(self) -> None:
        forbidden = ("/Users/", "/root/", "/tmp/", "reference-batch")
        for path in sorted(DATA.rglob("*.json")):
            for text in strings(json.loads(path.read_text(encoding="utf-8"))):
                self.assertFalse(
                    any(marker in text for marker in forbidden),
                    f"{path.relative_to(DATA)}: {text}",
                )

    def _body(self, case_id: str) -> dict[str, Any]:
        for call in self.calls["calls"]:
            if call["case"] == case_id:
                return call["response"]["body"]
        raise AssertionError(f"no recorded call for {case_id}")


def strings(value: Any):
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for item in value.values():
            yield from strings(item)
    elif isinstance(value, list):
        for item in value:
            yield from strings(item)


if __name__ == "__main__":
    unittest.main()
