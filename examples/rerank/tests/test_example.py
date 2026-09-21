"""Tamper tests for the offline checks.

    python3 fetch.py
    python3 -m unittest discover -s tests -v

Standard library only, and no server. Each one asserts that a specific forgery
fails closed.

A missing `data/` directory fails these tests rather than skipping them. A check
that cannot run has not passed.
"""

from __future__ import annotations

import contextlib
import copy
import hashlib
import importlib.util
import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
REQUIRED = ("inputs/cases.json", "calls.json", "manifest.json")


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


class RerankExampleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        missing = [name for name in REQUIRED if not (DATA / name).exists()]
        if missing:
            raise AssertionError(f"fetch the evidence first (python3 fetch.py); missing {', '.join(missing)}")
        cls.payload = json.loads((DATA / "inputs/cases.json").read_text(encoding="utf-8"))
        cls.calls = json.loads((DATA / "calls.json").read_text(encoding="utf-8"))
        cls.manifest = json.loads((DATA / "manifest.json").read_text(encoding="utf-8"))

    def test_manifest_pins_every_fetched_file(self) -> None:
        """The digest chain fetch.py walked, checked again after the download.

        fetch.py pins manifest.json by the digest in its own source and then
        pins every other file by the digest inside that manifest, so the bytes
        are authenticated before anything parses them. A rewritten candidate
        excerpt is caught here, at fetch time, rather than by a digest recorded
        beside the text it covers.
        """
        pinned = self.manifest["files_sha256"]
        self.assertEqual(pinned["calls.json"], self.manifest["calls_sha256"])
        on_disk = {
            str(path.relative_to(DATA)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(DATA.rglob("*"))
            if path.is_file() and path.name not in {"manifest.json", ".sie-evidence"}
        }
        self.assertEqual(on_disk, pinned)

    def test_corpus_is_the_shape_the_published_figures_assume(self) -> None:
        cases = self.payload["cases"]
        self.assertEqual(len(cases), 24)
        self.assertEqual(len({case["id"] for case in cases}), 24)
        self.assertEqual(len({case["agency"] for case in cases}), 24)
        for case in cases:
            self.assertEqual(len(case["candidates"]), 4)
            roles = sorted(candidate["role"] for candidate in case["candidates"])
            self.assertEqual(roles, ["asked-about-final", "asked-about-proposal",
                                     "neighbour-final", "neighbour-proposal"])
            # The query is the shared docket title. It must not repeat the
            # neighbour's, or the pool would not be competitive on subject.
            self.assertNotEqual(case["query"], case["neighbour_title"])
            for candidate in case["candidates"]:
                expected = "Rule" if candidate["role"].endswith("final") else "Proposed Rule"
                self.assertEqual(candidate["type"], expected, candidate["id"])

    def test_recorded_envelopes_match_the_runner(self) -> None:
        self.assertEqual(run.check(DATA), 0)

    def test_published_figures_match_the_recording(self) -> None:
        self.assertEqual(score.main_with(DATA), 0)

    def test_a_missing_call_fails_rather_than_shrinking_the_denominator(self) -> None:
        """Absent data is a failure, never a skip."""
        payload = copy.deepcopy(self.calls)
        dropped = f"{self.payload['cases'][0]['id']}/proposed-positive"
        payload["calls"] = [call for call in payload["calls"] if call["id"] != dropped]
        payload["call_count"] = len(payload["calls"])
        with _temp_calls(payload) as data_dir:
            self.assertEqual(score.main_with(data_dir), 1)

    def test_two_responses_swapped_between_arms_fails_closed(self) -> None:
        """The forgery every count and digest survives.

        Trading two responses between arms of the same case keeps the call
        count, the rank set and every rebuilt request envelope intact, so
        `run.py --check` still passes. Only the figures pinned in score.py's
        own source refuse it.
        """
        payload = copy.deepcopy(self.calls)
        case_id = self.payload["cases"][0]["id"]
        by_id = {call["id"]: call for call in payload["calls"]}
        left, right = by_id[f"{case_id}/none"], by_id[f"{case_id}/proposed-positive"]
        left["response"], right["response"] = right["response"], left["response"]
        with _temp_calls(payload) as data_dir:
            self.assertEqual(run.check(data_dir), 0, "the envelopes are untouched, so --check must still pass")
            self.assertEqual(score.main_with(data_dir), 1)

    def test_a_candidate_scored_twice_fails_closed(self) -> None:
        """A count is not a set.

        Replacing one scored row with a duplicate of another keeps the row
        count and the 0..n-1 ranks intact, so only comparing identities catches
        it.
        """
        payload = copy.deepcopy(self.calls)
        case_id = self.payload["cases"][0]["id"]
        body = next(c for c in payload["calls"] if c["id"] == f"{case_id}/none")["response"]["body"]
        body["scores"][0]["item_id"] = body["scores"][1]["item_id"]
        with _temp_calls(payload) as data_dir:
            self.assertEqual(score.main_with(data_dir), 1)

    def test_a_missing_manifest_fails_rather_than_skipping_its_own_check(self) -> None:
        """The revision check must not vanish with the file it reads.

        score.py used to load the manifest only `if manifest_path.exists()`,
        so deleting one file removed the check that the recording came from the
        published weights, and a recording from any other checkpoint scored
        clean. Missing evidence is a failure, not a licence to skip the check
        over it.
        """
        with _temp_calls(self.calls) as data_dir:
            (data_dir / "manifest.json").unlink()
            with self.assertRaisesRegex(SystemExit, "manifest.json is missing"):
                score.main_with(data_dir)

    def test_a_call_from_another_deployment_fails_closed(self) -> None:
        """Every call has to come from the deployment the manifest names.

        Checked against the manifest's `deployment_revision` and NOT against
        MODEL_REVISION: the SDK's `last_model_revision` is the SIE deployment
        digest that models served together share, not the HuggingFace revision
        of these weights. Comparing it with MODEL_REVISION would reject all 120
        of this example's own calls.
        """
        manifest = copy.deepcopy(self.manifest)
        self.assertNotEqual(manifest["deployment_revision"], manifest["model_revision"])
        for call in self.calls["calls"]:
            self.assertEqual(call["recorded"]["model_revision"], manifest["deployment_revision"])
        manifest["deployment_revision"] = "0" * 64
        with _temp_calls(self.calls, manifest=manifest) as data_dir:
            self.assertEqual(score.main_with(data_dir), 1)

    def test_an_existing_backup_path_is_refused_not_deleted(self) -> None:
        """A path this script names is not a path it owns.

        fetch.py moves `data/` aside before renaming the new tree into place.
        It used to rmtree whatever already sat at that name, which would delete
        a reader's unrelated directory without a word.
        """
        fetch = _module("fetch")
        with tempfile.TemporaryDirectory() as tmp:
            dest = Path(tmp) / "data"
            dest.mkdir()
            backup = fetch.backup_path(dest)
            backup.mkdir()
            (backup / "unrelated.txt").write_text("not ours")

            self.assertIn("already exists", fetch.refuse_reason(dest) or "")
            with self.assertRaisesRegex(RuntimeError, "backup path"):
                fetch.swap_into_place(Path(tmp) / "staging", dest)
            self.assertTrue((backup / "unrelated.txt").exists())

    def test_a_recording_from_another_model_revision_is_not_scored(self) -> None:
        manifest = copy.deepcopy(self.manifest)
        manifest["model_revision"] = "0" * 40
        with _temp_calls(self.calls, manifest=manifest) as data_dir:
            with self.assertRaisesRegex(SystemExit, "model revision"):
                score.main_with(data_dir)

    def test_metadata_has_no_temporary_filesystem_paths(self) -> None:
        forbidden = ("/Users/", "/root/", "/tmp/", "/private/tmp", "reference-batch")
        for path in sorted(DATA.rglob("*.json")):
            for text in strings(json.loads(path.read_text(encoding="utf-8"))):
                self.assertFalse(
                    any(marker in text for marker in forbidden),
                    f"{path.relative_to(DATA)}: {text}",
                )

    def test_every_candidate_cites_a_federal_register_url(self) -> None:
        for case in self.payload["cases"]:
            for candidate in case["candidates"]:
                self.assertTrue(
                    candidate["html_url"].startswith("https://www.federalregister.gov/documents/"),
                    candidate["html_url"],
                )


@contextlib.contextmanager
def _temp_calls(calls: dict[str, Any], manifest: dict[str, Any] | None = None):
    """A copy of data/ with calls.json, and optionally manifest.json, replaced."""
    with tempfile.TemporaryDirectory() as tmp:
        data_dir = Path(tmp) / "data"
        shutil.copytree(DATA, data_dir)
        (data_dir / "calls.json").write_text(json.dumps(calls, indent=1, ensure_ascii=False) + "\n",
                                             encoding="utf-8")
        if manifest is not None:
            (data_dir / "manifest.json").write_text(json.dumps(manifest, indent=1, ensure_ascii=False) + "\n",
                                                    encoding="utf-8")
        yield data_dir


if __name__ == "__main__":
    unittest.main()
