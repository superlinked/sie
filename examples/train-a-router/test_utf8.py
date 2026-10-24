"""Offline Unicode regressions for the reader boundary around frozen recipes."""

import hashlib
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SAMPLE = "Linguini: ə ʃ ŋ — 日本語"
ASCII_ENV = {
    **os.environ,
    "LC_ALL": "C",
    "LANG": "C",
    "PYTHONCOERCECLOCALE": "0",
    "PYTHONUTF8": "0",
    "PYTHONPATH": str(ROOT),
}


def run_python(*arguments):
    return subprocess.run(
        [sys.executable, "-X", "utf8=0", *map(str, arguments)],
        cwd=ROOT,
        env=ASCII_ENV,
        capture_output=True,
        timeout=30,
        check=False,
    )


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes((json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode("utf-8"))


class Utf8ReaderTests(unittest.TestCase):
    def assert_success(self, result):
        self.assertEqual(result.returncode, 0, result.stderr.decode("utf-8", errors="replace"))

    def probe(self, root, *, bootstrap):
        path = root / "probe.py"
        path.write_text(
            "import json, sys\n"
            "from pathlib import Path\n"
            "from reader_runtime import ensure_utf8\n"
            "from embedding import recipes\n"
            "from native import recipe\n"
            + ("ensure_utf8()\n" if bootstrap else "")
            + "output, request = Path(sys.argv[1]), sys.argv[2]\n"
            "value = {'request': request}\n"
            "for name, module in [('embedding', recipes), ('native', recipe)]:\n"
            "    path = output / (name + '.json')\n"
            "    module.write(path, value)\n"
            "    assert module.read(path) == value\n"
            "print(json.dumps({'utf8': sys.flags.utf8_mode, 'options': sys._xoptions,\n"
            "                  'warnings': sys.warnoptions, 'args': sys.argv[1:],\n"
            "                  'unbuffered': sys.stdout.write_through}))\n",
            encoding="utf-8",
        )
        return path

    def assert_probe_bytes(self, output):
        expected = (json.dumps({"request": SAMPLE}, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode()
        for name in ("embedding", "native"):
            self.assertEqual((output / f"{name}.json").read_bytes(), expected)

    def test_ascii_parent_reproduces_frozen_reader_and_writer_failure(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "sample.json"
            write_json(path, {"request": SAMPLE})
            result = run_python(
                "-c",
                "import codecs, locale, sys\n"
                "from native import recipe\n"
                "assert sys.flags.utf8_mode == 0\n"
                "assert codecs.lookup(locale.getpreferredencoding(False)).name == 'ascii'\n"
                "try:\n"
                "    recipe.read(sys.argv[1])\n"
                "except UnicodeDecodeError:\n"
                "    pass\n"
                "else:\n"
                "    raise AssertionError('Expected an ASCII decode failure')\n"
                "try:\n"
                "    recipe.write(sys.argv[2], {'request': '\\u0259 \\u0283 \\u014b'})\n"
                "except UnicodeEncodeError:\n"
                "    pass\n"
                "else:\n"
                "    raise AssertionError('Expected an ASCII encode failure')\n",
                path,
                path.with_name("failed-write.json"),
            )
            self.assert_success(result)

    def test_startup_preserves_flags_unicode_arguments_and_utf8_bytes(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = root / "ə output"
            result = run_python(
                "-u", "-W", "error", "-X", "reader_test=retained", self.probe(root, bootstrap=True), output, SAMPLE
            )
            self.assert_success(result)
            receipt = json.loads(result.stdout)
            self.assertEqual(receipt["utf8"], 1)
            self.assertEqual(receipt["options"]["reader_test"], "retained")
            self.assertIn("error", receipt["warnings"])
            self.assertTrue(receipt["unbuffered"])
            self.assertEqual(receipt["args"], [str(output), SAMPLE])
            self.assert_probe_bytes(output)

    def test_workflow_child_uses_utf8_even_when_parent_does_not(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = root / "result"
            result = run_python(
                "-c",
                "import sys, workflow\nassert sys.flags.utf8_mode == 0\nworkflow.invoke(*sys.argv[1:])\n",
                self.probe(root, bootstrap=False),
                output,
                SAMPLE,
            )
            self.assert_success(result)
            self.assertEqual(json.loads(result.stdout)["utf8"], 1)
            self.assert_probe_bytes(output)

    def test_actual_cli_reads_unicode_with_file_module_and_separator_invocations(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            candidate = {
                "candidate_id": 0,
                "policy": {"mode": "always_strong"},
                "calibration": {
                    "answer_accuracy_percent": 80,
                    "gap_vs_strong_pp": 0,
                    "answer_cost_usd_per10000": 1,
                    "total_cost_usd_per10000": 1,
                    "cheap_share": 0,
                },
            }
            write_json(
                root / "calibration-frontiers.json",
                {
                    "format": "bbeh-calibration-frontiers-v1",
                    "final_outcomes_accessed": False,
                    "ids": ["synthetic"],
                    "methods": {
                        SAMPLE: {
                            "no_preferred_quality_target": True,
                            "metadata_assisted": False,
                            "candidates": [candidate],
                            "frontier": [candidate],
                        }
                    },
                },
            )
            for index, command in enumerate(((ROOT / "router.py",), ("-m", "router"), ("--", ROOT / "router.py"))):
                with self.subTest(command=command):
                    output = root / f"ə view {index}.html"
                    result = run_python(*command, "frontier", "--run", root, "--output", output)
                    self.assert_success(result)
                    self.assertEqual(json.loads(result.stdout)["open"], str(output))
                    self.assertIn(SAMPLE.encode("utf-8"), output.read_bytes())

    def test_cli_retains_argument_error_exit_status(self):
        result = run_python(ROOT / "router.py", "--invalid-reader-option")
        self.assertEqual(result.returncode, 2)
        self.assertIn(b"usage:", result.stderr)

    def test_frozen_source_bytes_still_match_the_manifest(self):
        manifest = json.loads((ROOT / "frozen-source-manifest.json").read_bytes())
        for name, expected in manifest["files"].items():
            with self.subTest(path=name):
                data = (ROOT / name).read_bytes()
                self.assertEqual(len(data), expected["bytes"])
                self.assertEqual(hashlib.sha256(data).hexdigest(), expected["sha256"])


if __name__ == "__main__":
    unittest.main()
