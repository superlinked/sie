import contextlib
import io
import json
import tempfile
import unittest
from collections import Counter
from pathlib import Path

import score

EXAMPLE_DIR = Path(__file__).resolve().parent.parent
VALIDATION_DIR = EXAMPLE_DIR / "validation"
REFERENCES_DIR = EXAMPLE_DIR / "references"
RESULT_KIND = "published-benchmark-reproduction"
REQUESTS_SUFFIX = ".requests.jsonl"
EXPECTED_RUNS = {
    "prod-web-fw-20260905-splade-nfcorpus-quality-02": 3,
    "prod-web-fw-20260905-gte-moderncolbert-scifact-perf-v2-r1": 3,
}


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def results():
    return sorted(VALIDATION_DIR.glob("*.json"))


def performance_results():
    return [path for path in results() if load(path)["evaluation_mode"] == score.PERFORMANCE_MODE]


def quality_results():
    return [path for path in results() if load(path)["evaluation_mode"] != score.PERFORMANCE_MODE]


def requests_log(result_path):
    return result_path.with_name(result_path.stem + REQUESTS_SUFFIX)


@contextlib.contextmanager
def away_from_the_recorded_paths():
    with tempfile.TemporaryDirectory() as temporary, contextlib.chdir(temporary):
        yield Path(temporary)


def performance_result(directory, requests_path, rows):
    score.write_jsonl(requests_path, rows)
    result_path = Path(directory) / "result.json"
    record = {
        "evaluation_mode": score.PERFORMANCE_MODE,
        "schedule": {"levels": [{"name": "fixed-1-rps", "rate": 1.0, "duration_s": 2.0}]},
        "run": {"request_count": len(rows), "requests_path": str(requests_path)},
        "comparison": [],
    }
    score.write_json(result_path, record)
    return result_path


def request_row(sequence, latency_ms):
    return {
        "sequence": sequence,
        "level_name": "fixed-1-rps",
        "offset_s": float(sequence),
        "actual_send_offset_ns": sequence * score.NS_PER_S,
        "completed_offset_ns": sequence * score.NS_PER_S + latency_ms * score.NS_PER_MS,
        "scheduling_lag_ms": 0.0,
        "status": "success",
        "attempt": 1,
        "client_retry_count": 0,
        "in_flight_at_send": 0,
    }


class ValidationDirectoryTest(unittest.TestCase):
    def test_holds_the_result_records_and_the_performance_requests_logs_only(self):
        files = sorted(VALIDATION_DIR.iterdir())
        self.assertTrue(files)
        for path in files:
            with self.subTest(path.name):
                if path.name.endswith(REQUESTS_SUFFIX):
                    self.assertTrue(path.with_name(path.name.removesuffix(REQUESTS_SUFFIX) + ".json").is_file())
                    self.assertTrue(score.read_jsonl(path))
                else:
                    self.assertEqual(path.suffix, ".json")
                    self.assertNotIn(".predictions", path.name)
                    load(path)
        self.assertEqual(Counter(load(path)["reference_id"] for path in results()), EXPECTED_RUNS)

    def test_every_result_names_its_reference_and_carries_its_metric_names(self):
        for path in results():
            with self.subTest(path.name):
                record = load(path)
                self.assertEqual(record["kind"], RESULT_KIND)
                reference_path = REFERENCES_DIR / f"{record['reference_id']}.json"
                self.assertTrue(reference_path.is_file())
                reference = load(reference_path)
                self.assertTrue(path.name.startswith(record["reference_id"] + "-"))
                self.assertEqual(record["evaluation_mode"], reference["evaluation_mode"])
                self.assertEqual(record["model"], reference["model"])
                self.assertEqual(record["profile"], reference["profile"])
                self.assertEqual(record["task_key"], reference["task_key"])
                names = [row["name"] for row in record["metrics"]]
                self.assertEqual(sorted(names), sorted(row["name"] for row in reference["metrics"]))
                self.assertEqual([row["name"] for row in record["comparison"]], names)

    def test_each_performance_result_has_its_requests_log_beside_it(self):
        performance = performance_results()
        self.assertTrue(performance)
        for path in performance:
            with self.subTest(path.name):
                record = load(path)
                log = requests_log(path)
                self.assertTrue(log.is_file())
                self.assertEqual(Path(record["run"]["requests_path"]).name, log.name)
                self.assertEqual(len(score.read_jsonl(log)), record["run"]["request_count"])


class RescoreTest(unittest.TestCase):
    def test_performance_results_re_aggregate_exactly_through_the_sibling_fallback(self):
        with away_from_the_recorded_paths():
            for path in performance_results():
                with self.subTest(path.name):
                    record, rows, comparison = score.rescore(path)
                    self.assertFalse(Path(record["run"]["requests_path"]).exists())
                    self.assertEqual(rows, record["metrics"])
                    self.assertEqual(comparison, record["comparison"])
                    self.assertEqual(len(rows), len(score.PERFORMANCE_METRICS))

    def test_quality_results_name_the_missing_predictions_file(self):
        with away_from_the_recorded_paths():
            for path in quality_results():
                with self.subTest(path.name):
                    with self.assertRaises(ValueError) as caught:
                        score.rescore(path)
                    message = str(caught.exception)
                    self.assertIn(Path(load(path)["run"]["predictions_path"]).name, message)
                    self.assertIn("the quality run has to be repeated", message)


class SiblingFallbackTest(unittest.TestCase):
    def test_the_recorded_path_is_used_while_it_exists(self):
        with tempfile.TemporaryDirectory() as temporary:
            recorded = Path(temporary) / "elsewhere" / "run.requests.jsonl"
            recorded.parent.mkdir()
            result_path = performance_result(temporary, recorded, [request_row(0, 100.0)])
            score.write_jsonl(requests_log(result_path.with_name("run.json")), [request_row(0, 900.0)])
            _, rows, _ = score.rescore(result_path)
            self.assertEqual(next(row["value"] for row in rows if row["name"] == "latency_p50_ms"), 100.0)

    def test_a_same_named_file_beside_the_result_replaces_a_missing_recorded_path(self):
        with tempfile.TemporaryDirectory() as temporary:
            recorded = Path(temporary) / "elsewhere" / "run.requests.jsonl"
            recorded.parent.mkdir()
            result_path = performance_result(temporary, recorded, [request_row(0, 100.0)])
            recorded.rename(result_path.with_name(recorded.name))
            _, rows, _ = score.rescore(result_path)
            self.assertEqual(next(row["value"] for row in rows if row["name"] == "latency_p50_ms"), 100.0)

    def test_a_missing_requests_file_is_named(self):
        with tempfile.TemporaryDirectory() as temporary:
            recorded = Path(temporary) / "elsewhere" / "run.requests.jsonl"
            recorded.parent.mkdir()
            result_path = performance_result(temporary, recorded, [request_row(0, 100.0)])
            recorded.unlink()
            with self.assertRaises(ValueError) as caught:
                score.rescore(result_path)
            self.assertIn("run.requests.jsonl", str(caught.exception))
            self.assertNotIn("quality run", str(caught.exception))
            with self.assertRaises(SystemExit) as exit_caught, contextlib.redirect_stderr(io.StringIO()):
                score.main([str(result_path)])
            self.assertEqual(exit_caught.exception.code, 2)


if __name__ == "__main__":
    unittest.main()
