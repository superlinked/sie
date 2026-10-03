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
import subprocess
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
    @classmethod
    def setUpClass(cls) -> None:
        missing = [name for name in REQUIRED if not (DATA / name).exists()]
        if missing:
            raise AssertionError(f"fetch the evidence first (python3 fetch.py); missing {', '.join(missing)}")
        cls.cases = json.loads((DATA / "inputs/cases.json").read_text(encoding="utf-8"))
        cls.sources = json.loads((DATA / "inputs/sources.json").read_text(encoding="utf-8"))
        cls.calls = json.loads((DATA / "calls.json").read_text(encoding="utf-8"))
        cls.manifest = json.loads((DATA / "manifest.json").read_text(encoding="utf-8"))

    def test_inputs_are_exact_primary_source_excerpts(self) -> None:
        score.verify_inputs(self.cases, self.sources)
        self.assertEqual(len(self.cases["cases"]), 4)

    def test_all_recorded_spans_match_the_source_offsets(self) -> None:
        total = 0
        for case_id, case in self.cases["cases"].items():
            spans = score.entities(case_id, case, self._body(case_id))
            total += len(spans)
            for span in spans:
                self.assertEqual(case["text"][span["start"] : span["end"]], span["text"])
        self.assertEqual(total, 53)

    def test_recorded_envelopes_match_the_runner(self) -> None:
        self.assertEqual(run.check(DATA), 0)

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

    def test_boolean_score_fails_closed(self) -> None:
        case_id = "scotus_two_contracts"
        body = copy.deepcopy(self._body(case_id))
        body["entities"][0]["score"] = True
        with self.assertRaisesRegex(SystemExit, "non-numeric score"):
            score.entities(case_id, self.cases["cases"][case_id], body)

    def test_non_finite_scores_fail_closed(self) -> None:
        case_id = "scotus_two_contracts"
        body = copy.deepcopy(self._body(case_id))
        body["entities"][0]["score"] = float("nan")
        with self.assertRaisesRegex(SystemExit, "non-numeric score"):
            score.entities(case_id, self.cases["cases"][case_id], body)

    def test_boolean_offsets_fail_closed(self) -> None:
        case_id = "scotus_two_contracts"
        body = copy.deepcopy(self._body(case_id))
        body["entities"][0]["start"] = True
        with self.assertRaisesRegex(SystemExit, "non-integer offsets"):
            score.entities(case_id, self.cases["cases"][case_id], body)

    def test_unrequested_label_fails_closed(self) -> None:
        case_id = "scotus_two_contracts"
        body = copy.deepcopy(self._body(case_id))
        body["entities"][0]["label"] = "sales region"
        with self.assertRaisesRegex(SystemExit, "unrequested label"):
            score.entities(case_id, self.cases["cases"][case_id], body)

    def test_moved_span_fails_closed(self) -> None:
        """An offset edited without the text it points at.

        The span still carries a requested label and a plausible score, so only
        re-reading input_text[start:end] catches it.
        """
        case_id = "ntsb_detector_alert"
        body = copy.deepcopy(self._body(case_id))
        body["entities"][0]["start"] += 1
        with self.assertRaisesRegex(SystemExit, "not the text at its own offsets"):
            score.entities(case_id, self.cases["cases"][case_id], body)

    def test_empty_response_fails_the_required_anchors(self) -> None:
        case_id = "sec_filing_amendment"
        case = self.cases["cases"][case_id]
        body = copy.deepcopy(self._body(case_id))
        body["entities"] = []
        spans = score.entities(case_id, case, body)
        observed = {score.anchor_key(span) for span in spans}
        matched = sum(1 for anchor in case["required_anchors"] if score.anchor_key(anchor) in observed)
        self.assertEqual(matched, 0)
        self.assertEqual(len(case["required_anchors"]), 8)

    def test_cms_false_positive_is_preserved_but_not_required(self) -> None:
        """The model's wrong answer stays in the recording.

        `proof of delivery` came back under `missing documentation` and the
        text does not say it is missing. Keeping it visible is the point, so
        this asserts both that it is still there and that nothing requires it.
        """
        case_id = "cms_lower_limb_orthosis"
        case = self.cases["cases"][case_id]
        spans = score.entities(case_id, case, self._body(case_id))
        wrong = [
            span for span in spans if span["label"] == "missing documentation" and span["text"] == "proof of delivery"
        ]
        self.assertEqual(len(wrong), 1)
        required = {score.anchor_key(anchor) for anchor in case["required_anchors"]}
        self.assertNotIn(score.anchor_key(wrong[0]), required)

    def test_coordinated_excerpt_mutation_fails_the_canonical_check(self) -> None:
        """The forgery a single declared digest cannot catch.

        Rewrite the text and the digest recorded beside it together and that
        pair agrees with itself. The canonical digest in sources.json is what
        refuses it, which is why both files travel together.
        """
        cases = copy.deepcopy(self.cases)
        case = cases["cases"]["scotus_two_contracts"]
        case["text"] += " tampered"
        case["source"]["text"] = case["text"]
        case["source"]["sha256"] = score.sha256_text(case["text"])
        with self.assertRaisesRegex(SystemExit, "canonical one in sources.json"):
            score.verify_inputs(cases, self.sources)

    def test_a_failed_run_is_never_scored(self) -> None:
        payload = copy.deepcopy(self.calls)
        payload["complete"] = False
        payload["failed_calls"] = 1
        with self.assertRaisesRegex(SystemExit, "not a complete run"):
            score.scored_calls(payload)

    def test_score_reproduces_the_published_figures(self) -> None:
        result = subprocess.run(  # noqa: S603
            [sys.executable, str(ROOT / "score.py"), "--data", str(DATA)],
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("53 spans returned across the 4 cases", result.stdout)
        self.assertIn("28 of 28 required anchors matched", result.stdout)

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


if __name__ == "__main__":
    unittest.main()
